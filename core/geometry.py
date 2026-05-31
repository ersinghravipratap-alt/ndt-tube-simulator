"""
Geometry Module

Defines geometric models for tubes and ultrasonic probes.
Handles coordinate transformations, probe positioning, and geometry validation.

COORDINATE SYSTEM:
==================
X-axis: Radial (tube circumference)
Y-axis: Axial (along tube length)
Z-axis: Through-thickness (radial direction)

Standard right-hand coordinate system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from .material_properties import Material


@dataclass
class Tube:
    """
    Cylindrical tube geometry for NDT inspection.
    
    Parameters:
    -----------
    outer_diameter : float
        Outer diameter in mm
    
    wall_thickness : float
        Wall thickness in mm
    
    length : float
        Tube length in mm (inspection region)
    
    material : Material
        Material properties object
    """
    
    outer_diameter: float
    wall_thickness: float
    length: float
    material: Material
    
    @property
    def inner_diameter(self) -> float:
        """Calculate inner diameter from OD and wall thickness."""
        return self.outer_diameter - 2 * self.wall_thickness
    
    @property
    def outer_radius(self) -> float:
        return self.outer_diameter / 2
    
    @property
    def inner_radius(self) -> float:
        return self.inner_diameter / 2
    
    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area of tube material (mm²)"""
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)
    
    def __repr__(self) -> str:
        return (
            f"Tube(OD={self.outer_diameter}mm, WT={self.wall_thickness}mm, "
            f"L={self.length}mm, Material={self.material.name})"
        )


@dataclass
class AngleBeamProbe:
    """
    Angle-beam ultrasonic probe for immersion testing.
    
    Models three focus types:
    1. SPOT FOCUS: Spherical lens - focuses to a point
    2. LINE FOCUS: Cylindrical lens - focuses to a line
    3. FLAT FOCUS: Flat piston - parallel beam
    
    Parameters:
    -----------
    frequency : float
        Ultrasonic frequency in MHz
    
    element_diameter : float
        Transducer element diameter in mm
    
    refraction_angle : float
        Refraction angle in water in degrees (0-90)
        - 0° = normal incidence (straight beam)
        - 45° = common angle for S-wave inspection
        - 70°+ = high-angle L-wave inspection
    
    focal_length : float
        Focal distance in mm (for spot and line focus)
        - None for flat focus
    
    focus_type : str
        'spot' - spherical lens (point focus)
        'line' - cylindrical lens (line focus)
        'flat' - flat piston (parallel beam)
    
    material_contact : Material
        Contact medium (typically water for immersion)
    
    PHYSICS OF ANGLE BEAMS:
    =======================
    When a plane wave hits a material interface at angle θ:
    1. Part reflects back into water at angle θ
    2. Part refracts into material using Snell's law:
       sin(θ_refracted) = (c_material / c_water) * sin(θ_incident)
    
    For Steel (c=5900 m/s) in Water (c=1480 m/s):
    - 45° incident → ~17° refracted (high refraction!)
    - 70° incident → approaches critical angle (~14.5°)
    - >80° incident → mode conversion to S-waves
    """
    
    frequency: float  # MHz
    element_diameter: float  # mm
    refraction_angle: float  # degrees
    focal_length: Optional[float] = None  # mm
    focus_type: str = 'spot'  # 'spot', 'line', or 'flat'
    material_contact: Material = None
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0 <= self.refraction_angle <= 90:
            raise ValueError("refraction_angle must be 0-90 degrees")
        
        if self.focus_type not in ['spot', 'line', 'flat']:
            raise ValueError("focus_type must be 'spot', 'line', or 'flat'")
        
        if self.focus_type != 'flat' and self.focal_length is None:
            raise ValueError(f"focal_length required for {self.focus_type} focus")
        
        if self.material_contact is None:
            self.material_contact = Material.from_preset('water')
    
    @property
    def wavelength(self) -> float:
        """
        Wavelength in contact medium at this frequency.
        
        Formula: λ = c / f
        
        where c=velocity, f=frequency
        
        Used to determine near-field/far-field behavior and beam divergence.
        """
        return self.material_contact.velocity_L / (self.frequency * 1e6)  # m → mm conversion
    
    @property
    def near_field_distance(self) -> float:
        """
        Near-field (Fresnel zone) distance in contact medium.
        
        Formula: N = (D² * f) / (4 * c)
        
        where D=element diameter, f=frequency, c=velocity
        
        PHYSICS NOTE:
        - Inside near-field: beam is complex, amplitude fluctuates
        - Beyond near-field: beam diverges (Fraunhofer zone)
        - For NDT: stay in near-field for sharp beam definition
        """
        D_m = self.element_diameter / 1000  # Convert mm to m
        f = self.frequency * 1e6  # Convert MHz to Hz
        c = self.material_contact.velocity_L
        
        return (D_m**2 * f) / (4 * c) * 1000  # Convert m to mm
    
    @property
    def beam_divergence_angle(self) -> float:
        """
        Half-angle beam divergence in far-field (Fraunhofer zone).
        
        Formula: θ_div = 0.866 * λ / D
        
        where λ=wavelength, D=element diameter
        
        PHYSICS NOTE:
        - Smaller diameter → larger divergence (wider beam)
        - Higher frequency → smaller wavelength → smaller divergence
        - Typical: 10mm element, 5 MHz: ~20° divergence
        """
        wavelength_mm = self.wavelength
        return np.degrees(0.866 * wavelength_mm / self.element_diameter)
    
    def get_focal_point_in_material(self, tube_material: Material,
                                     water_path: float) -> Tuple[float, float]:
        """
        Calculate where the focused beam converges inside the tube material.
        
        Accounts for refraction at water-steel interface using Snell's law.
        
        Parameters:
        -----------
        tube_material : Material
            Material of tube (typically steel)
        
        water_path : float
            Distance from probe face to tube surface in mm
        
        Returns:
        --------
        Tuple[float, float] :
            (axial_distance, radial_depth) in mm
            Focal point location relative to tube surface
        
        PHYSICS:
        1. Beam travels through water at angle θ_water
        2. At steel surface, refraction occurs: sin(θ_steel) = (c_steel/c_water)*sin(θ_water)
        3. Focal point calculated by tracing refracted ray to focus distance
        """
        # Snell's law: refracted angle in steel
        sin_theta_water = np.sin(np.radians(self.refraction_angle))
        sin_theta_steel = (tube_material.velocity_L / 
                          self.material_contact.velocity_L) * sin_theta_water
        
        if sin_theta_steel > 1.0:
            # Total internal reflection - no refraction
            theta_steel = None
        else:
            theta_steel = np.degrees(np.arcsin(sin_theta_steel))
        
        # Axial distance traveled at angle theta_steel
        if theta_steel is not None and self.focal_length is not None:
            axial = self.focal_length * np.tan(np.radians(theta_steel))
            radial = self.focal_length / np.cos(np.radians(theta_steel))
        else:
            axial = 0
            radial = water_path
        
        return axial, radial
    
    @property
    def element_area(self) -> float:
        """Active element area in mm²"""
        if self.focus_type == 'line':
            # Cylindrical lens: area = diameter × length (approximate)
            return self.element_diameter ** 2
        else:
            # Circular element
            return np.pi * (self.element_diameter / 2) ** 2
    
    def __repr__(self) -> str:
        focus_str = f"{self.focus_type.upper()}"
        if self.focal_length is not None:
            focus_str += f" (f={self.focal_length}mm)"
        
        return (
            f"AngleBeamProbe(\n"
            f"  Frequency: {self.frequency} MHz\n"
            f"  Element: {self.element_diameter}mm diameter\n"
            f"  Refraction angle: {self.refraction_angle}°\n"
            f"  Focus: {focus_str}\n"
            f"  Wavelength: {self.wavelength:.3f}mm\n"
            f"  Near-field: {self.near_field_distance:.1f}mm\n"
            f"  Beam divergence: ±{self.beam_divergence_angle:.1f}°\n"
            f"  Contact medium: {self.material_contact.name}\n"
            f")"
        )
    
    def spot_focus_aperture(self, distance_from_focal_point: float) -> float:
        """
        Calculate beam diameter at given distance from focal point.
        
        Uses Gaussian beam approximation for focused probes.
        
        Parameters:
        -----------
        distance_from_focal_point : float
            Distance from focal point in mm (+/-). Negative = before focus,
            positive = after focus
        
        Returns:
        --------
        float : Beam diameter at that distance in mm
        
        PHYSICS:
        Focused beams create "focal zones" - near focus the beam is narrow,
        far from focus it diverges. This is modeled by:
        
        W(z) = W0 * sqrt(1 + (z/z_R)²)
        
        where:
        - W(z) = beam radius at distance z
        - W0 = beam radius at focal point
        - z_R = Rayleigh range (depth of field)
        """
        if self.focus_type == 'flat':
            # Flat transducers don't have focal point - constant beam
            return self.element_diameter
        
        # Rayleigh range (depth of field parameter)
        z_R = self.focal_length
        
        # Beam radius at focal point
        w0 = self.element_diameter / 4  # Approximately
        
        # Gaussian beam expansion
        w_z = w0 * np.sqrt(1 + (distance_from_focal_point / z_R) ** 2)
        
        return 2 * w_z  # Diameter = 2 * radius
    
    @staticmethod
    def create_immersion_probe(frequency: float,
                               element_diameter: float,
                               refraction_angle: float,
                               focus_type: str = 'spot',
                               focal_length: float = 50.0) -> 'AngleBeamProbe':
        """
        Factory method to create standard immersion probe.
        
        Parameters match common commercial NDT probes.
        """
        water = Material.from_preset('water')
        return AngleBeamProbe(
            frequency=frequency,
            element_diameter=element_diameter,
            refraction_angle=refraction_angle,
            focal_length=focal_length,
            focus_type=focus_type,
            material_contact=water
        )


if __name__ == '__main__':
    """
    Demonstration of geometry calculations.
    """
    print("=" * 70)
    print("GEOMETRY DEMONSTRATION")
    print("=" * 70)
    
    # Create tube
    steel = Material.from_preset('carbon_steel')
    tube = Tube(
        outer_diameter=50.0,
        wall_thickness=5.0,
        length=500.0,
        material=steel
    )
    
    print("\n1. TUBE GEOMETRY")
    print("-" * 70)
    print(f"{tube}")
    print(f"Inner diameter: {tube.inner_diameter:.1f} mm")
    print(f"Cross-sectional area: {tube.cross_sectional_area:.1f} mm²")
    
    # Create angle-beam probe
    probe = AngleBeamProbe.create_immersion_probe(
        frequency=5.0,
        element_diameter=10.0,
        refraction_angle=45.0,
        focus_type='spot',
        focal_length=50.0
    )
    
    print("\n2. ANGLE-BEAM PROBE")
    print("-" * 70)
    print(probe)
    
    print("\n3. FOCAL POINT LOCATION")
    print("-" * 70)
    water_path = 30.0  # mm
    axial, radial = probe.get_focal_point_in_material(steel, water_path)
    print(f"Water path: {water_path} mm")
    print(f"Focal point: {axial:.1f}mm axial, {radial:.1f}mm radial depth")
    
    print("\n4. BEAM APERTURE AT VARIOUS DISTANCES")
    print("-" * 70)
    for dist in [-20, -10, 0, 10, 20]:  # mm from focal point
        diameter = probe.spot_focus_aperture(dist)
        print(f"Distance {dist:+3d}mm from focal point: beam ∅ = {diameter:.2f}mm")
