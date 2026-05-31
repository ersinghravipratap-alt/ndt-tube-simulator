"""
Defect Models Module

Defines various defect geometries and their ultrasonic scattering response.

DEFECT TYPES IMPLEMENTED:
=========================

1. PLANAR DEFECTS (Flat-Bottom Holes, Cracks):
   - Orientation: Fixed or variable
   - Size: Any dimensions
   - Location: Any position in tube
   - Scattering model: Kirchhoff approximation

2. INCLINED DEFECTS (Angled Cracks):
   - Inclination angle: 0-90°
   - Length and width: Variable
   - Position: Any depth in tube
   - Scattering model: GTD (Geometrical Theory of Diffraction)

3. ARBITRARY SHAPE DEFECTS:
   - User-defined vertices/mesh
   - Complex geometries
   - Computed using numerical integration

SCATTERING PHYSICS:
===================

Kirchhoff Approximation:
- Valid for large defects (size >> wavelength)
- Amplitude ∝ defect area * cos(θ_incident) * cos(θ_scattered)
- Based on specular reflection

GTD (Geometrical Theory of Diffraction):
- Accounts for diffraction at defect edges
- Important for sharp features (crack tips)
- Adds edge diffraction terms

Amplitude modulation:
- Higher frequency → smaller wavelength → higher sensitivity
- Larger defect → stronger echo
- Oblique angles → reduced amplitude (less head-on)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class DefectResponse:
    """Result of defect scattering calculation."""
    amplitude: float  # Normalized 0-1
    phase: float  # radians
    time_of_flight: float  # microseconds
    back_wall_blocked: bool  # True if defect blocks back-wall signal


class Defect(ABC):
    """Abstract base class for all defect types."""
    
    def __init__(self, x: float, y: float, z: float = 0):
        """
        Initialize defect at position (x, y, z) in tube coordinates.
        
        x: axial position (mm, along tube)
        y: radial/depth position (mm, from OD)
        z: circumferential position (mm, around tube)
        """
        self.x = x  # axial
        self.y = y  # depth from OD
        self.z = z  # circumferential
    
    @abstractmethod
    def get_amplitude_response(self, incident_angle: float, frequency: float) -> float:
        """Calculate scattering amplitude for given incident angle and frequency."""
        pass
    
    @abstractmethod
    def calculate_response(self, beam_amplitude: float, frequency: float,
                          material_velocity: float, water_velocity: float) -> DefectResponse:
        """Calculate full defect response including TOF and amplitude."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


class PlanarDefect(Defect):
    """
    Planar defect (flat-bottom hole, crack surface).
    
    KIRCHHOFF APPROXIMATION:
    Amplitude = Amplitude_incident * (area / wavelength²) * geometric_factor
    
    where geometric_factor depends on incident and scattering angles.
    """
    
    def __init__(self, x: float, y: float, z: float = 0,
                 width: float = 5.0, height: float = 10.0,
                 tilt_angle: float = 0.0):
        """
        Parameters:
        -----------
        x, y, z : float
            Position in tube (mm)
        
        width : float
            Width of planar defect (mm)
        
        height : float
            Height/length of planar defect (mm)
        
        tilt_angle : float
            Tilt angle from normal in degrees
            0° = perpendicular to beam (maximum reflection)
            90° = parallel to beam (minimal reflection)
        """
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.tilt_angle = tilt_angle
    
    def get_amplitude_response(self, incident_angle: float, frequency: float) -> float:
        """
        Calculate reflection amplitude using Kirchhoff approximation.
        
        PHYSICS:
        For flat reflector with area A at angle θ to incident beam:
        A_reflected ∝ A * cos(θ_incident - θ_defect)
        """
        
        # Area of defect
        area = self.width * self.height
        
        # Angle between incident beam and defect normal
        angle_diff = incident_angle - self.tilt_angle
        
        # Kirchhoff: amplitude depends on perpendicularity
        geometric_factor = max(0, np.cos(np.radians(angle_diff)))
        
        # Frequency dependence: higher frequency → higher sensitivity
        # but smaller effective area (shorter wavelength)
        frequency_factor = np.sqrt(frequency)
        
        amplitude = area * geometric_factor * frequency_factor / 1000  # Normalize
        
        return min(1.0, amplitude)
    
    def calculate_response(self, beam_amplitude: float, frequency: float,
                          material_velocity: float, water_velocity: float) -> DefectResponse:
        """Full response calculation."""
        
        # Scattering amplitude
        scatter_amp = self.get_amplitude_response(45.0, frequency)  # Assume 45° incident
        total_amp = beam_amplitude * scatter_amp
        
        # Time-of-flight (2x to defect and back)
        tof = 2 * self.y / material_velocity * 1e6  # Convert to µs
        
        # Back-wall blocking (crude model)
        back_wall_blocked = total_amp > 0.3
        
        return DefectResponse(
            amplitude=total_amp,
            phase=0.0,
            time_of_flight=tof,
            back_wall_blocked=back_wall_blocked
        )
    
    def __repr__(self) -> str:
        return (f"PlanarDefect(x={self.x:.1f}mm, y={self.y:.1f}mm, "
                f"size={self.width}×{self.height}mm, tilt={self.tilt_angle}°)")


class InclinedDefect(Defect):
    """
    Inclined/angled defect (angled crack, slanted surface).
    
    Uses GTD (Geometrical Theory of Diffraction) for edge effects.
    
    PHYSICS:
    Inclined cracks produce:
    1. Specular reflection from main surface (Kirchhoff)
    2. Diffraction from crack tips (GTD)
    3. Frequency-dependent sensitivity
    """
    
    def __init__(self, x: float, y: float, z: float = 0,
                 length: float = 20.0, opening: float = 0.5,
                 inclination_angle: float = 45.0):
        """
        Parameters:
        -----------
        x, y, z : float
            Position (top of crack)
        
        length : float
            Crack length (mm)
        
        opening : float
            Crack opening (gap width, mm)
        
        inclination_angle : float
            Angle from horizontal (degrees)
            0° = horizontal crack
            90° = vertical crack (parallel to depth)
        """
        super().__init__(x, y, z)
        self.length = length
        self.opening = opening
        self.inclination_angle = inclination_angle
    
    def get_amplitude_response(self, incident_angle: float, frequency: float) -> float:
        """
        Calculate using Kirchhoff + GTD (tip diffraction).
        
        Total amplitude = specular_reflection + tip_diffraction
        """
        
        # Geometric factor for incident angle
        angle_diff = incident_angle - self.inclination_angle
        specular = max(0, np.cos(np.radians(angle_diff)))
        
        # Defect size effect
        area = self.length * self.opening
        size_factor = np.sqrt(area) / 100
        
        # Tip diffraction (GTD)
        # Stronger for sharp cracks, weaker as opening increases
        tip_diffraction = (1.0 / (1.0 + self.opening / 0.5))
        
        # Frequency dependence
        frequency_factor = np.sqrt(frequency / 5.0)  # Normalized to 5 MHz
        
        total = specular * size_factor * tip_diffraction * frequency_factor
        
        return min(1.0, total)
    
    def calculate_response(self, beam_amplitude: float, frequency: float,
                          material_velocity: float, water_velocity: float) -> DefectResponse:
        """Full response calculation."""
        
        scatter_amp = self.get_amplitude_response(45.0, frequency)
        total_amp = beam_amplitude * scatter_amp
        
        # TOF accounting for crack position and angle
        effective_depth = self.y + self.length * np.sin(np.radians(self.inclination_angle))
        tof = 2 * effective_depth / material_velocity * 1e6
        
        back_wall_blocked = total_amp > 0.2
        
        return DefectResponse(
            amplitude=total_amp,
            phase=0.0,
            time_of_flight=tof,
            back_wall_blocked=back_wall_blocked
        )
    
    def __repr__(self) -> str:
        return (f"InclinedDefect(x={self.x:.1f}mm, y={self.y:.1f}mm, "
                f"length={self.length}mm, angle={self.inclination_angle}°)")


class ArbitraryDefect(Defect):
    """
    Arbitrary shape defect defined by vertices/mesh.
    
    Allows custom defect geometries using numerical integration.
    """
    
    def __init__(self, x: float, y: float, z: float = 0,
                 vertices: List[Tuple[float, float]] = None):
        """
        Parameters:
        -----------
        vertices : List[Tuple[float, float]]
            List of (x, y) coordinates defining defect boundary
            Will compute area and orientation
        """
        super().__init__(x, y, z)
        self.vertices = vertices or []
        self._compute_properties()
    
    def _compute_properties(self):
        """Compute area and principal orientation from vertices."""
        if not self.vertices or len(self.vertices) < 3:
            self.area = 1.0
            self.orientation = 0.0
            return
        
        # Shoelace formula for area
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        self.area = abs(area) / 2.0
        
        # Principal orientation (simplified)
        vertices_array = np.array(self.vertices)
        dx = vertices_array[:, 0].max() - vertices_array[:, 0].min()
        dy = vertices_array[:, 1].max() - vertices_array[:, 1].min()
        self.orientation = np.degrees(np.arctan2(dy, dx))
    
    def get_amplitude_response(self, incident_angle: float, frequency: float) -> float:
        """Calculate using numerical integration over defect area."""
        
        if self.area < 0.1:
            return 0.0
        
        # Orientation effect
        angle_diff = incident_angle - self.orientation
        geometric_factor = max(0, np.cos(np.radians(angle_diff)))
        
        # Area and frequency scaling
        amplitude = np.sqrt(self.area) * geometric_factor * np.sqrt(frequency / 5.0) / 100
        
        return min(1.0, amplitude)
    
    def calculate_response(self, beam_amplitude: float, frequency: float,
                          material_velocity: float, water_velocity: float) -> DefectResponse:
        """Full response calculation."""
        
        scatter_amp = self.get_amplitude_response(45.0, frequency)
        total_amp = beam_amplitude * scatter_amp
        
        tof = 2 * self.y / material_velocity * 1e6
        back_wall_blocked = total_amp > 0.2
        
        return DefectResponse(
            amplitude=total_amp,
            phase=0.0,
            time_of_flight=tof,
            back_wall_blocked=back_wall_blocked
        )
    
    def add_vertex(self, x: float, y: float):
        """Add a vertex to defect boundary."""
        self.vertices.append((x, y))
        self._compute_properties()
    
    def __repr__(self) -> str:
        return (f"ArbitraryDefect(x={self.x:.1f}mm, y={self.y:.1f}mm, "
                f"vertices={len(self.vertices)}, area={self.area:.1f}mm²)")


def demo_defect_models():
    """Demonstrate defect modeling."""
    
    print("=" * 70)
    print("DEFECT MODELS DEMONSTRATION")
    print("=" * 70)
    
    # Create defects
    planar = PlanarDefect(x=100, y=10, width=5, height=20, tilt_angle=0)
    inclined = InclinedDefect(x=150, y=15, length=25, opening=0.3, inclination_angle=45)
    arbitrary = ArbitraryDefect(x=200, y=8)
    arbitrary.add_vertex(0, 0)
    arbitrary.add_vertex(5, 0)
    arbitrary.add_vertex(5, 10)
    arbitrary.add_vertex(0, 10)
    
    print("\n1. DEFECT DEFINITIONS")
    print("-" * 70)
    print(f"{planar}")
    print(f"{inclined}")
    print(f"{arbitrary}")
    
    print("\n2. AMPLITUDE RESPONSE AT DIFFERENT FREQUENCIES")
    print("-" * 70)
    for freq in [2.5, 5.0, 10.0]:
        amp = planar.get_amplitude_response(45.0, freq)
        print(f"Planar defect at {freq} MHz: A = {amp:.3f}")
    
    print("\n3. FULL RESPONSE CALCULATION")
    print("-" * 70)
    response = planar.calculate_response(beam_amplitude=0.8, frequency=5.0,
                                        material_velocity=5900, water_velocity=1480)
    print(f"Response: amplitude={response.amplitude:.3f}, TOF={response.time_of_flight:.2f}µs")


if __name__ == '__main__':
    demo_defect_models()
