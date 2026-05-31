# ndt-tube-simulator
for immersion UT simulations
# NDT Tube Simulator
## Advanced Angle-Beam Immersion Ultrasonic Testing Simulator

A comprehensive Python-based Non-Destructive Testing (NDT) simulator specifically designed for **angle-beam immersion ultrasonic testing (UT)** of tubes with advanced features like mode conversion, multi-type probe geometries, and complex defect modeling.

### 🎯 Features

- **Angle-Beam Immersion Inspection**: Full control over probe angles, water path, and tube geometry
- **Mode Conversion**: Automatic L-wave to S-wave conversion at material interfaces
- **Multiple Probe Types**:
  - Spot-focused probes (spherical lens focusing)
  - Line-focused probes (cylindrical lens)
  - Flat-focused probes (flat piston transducers)
- **Complex Defect Modeling**:
  - Planar flaws (cracks, laps)
  - Inclined defects (angled cracks, corrosion)
  - Arbitrary shape defects (mesh-based modeling)
- **Physics-Based Simulation**:
  - Semi-analytical beam computation (Pencil Method + Huygens' Principle)
  - Kirchhoff approximation for defect scattering
  - GTD (Geometrical Theory of Diffraction) for crack-tip diffraction
  - Full mode conversion tracking
- **Visualization**:
  - A-scan (amplitude vs. time)
  - B-scan (cross-sectional imaging)
  - C-scan (plan view)
  - 3D beam propagation and defect interaction

### 📁 Project Structure
