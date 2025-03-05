# CLAUDE.md - MobiusTorus Project Guide

## Run Commands
- Run visualization: `python mobius.py`
- No formal test suite is present in this repository

## Code Style Guidelines
- **Imports**: Standard Python style with major libraries at the top (numpy, matplotlib)
- **Formatting**: 4-space indentation, PEP 8 compliant
- **Naming Conventions**:
  - Functions: snake_case (e.g., plot_polygon_torus)
  - Variables: snake_case (e.g., twist_multiplier)
  - Constants: UPPER_CASE (none present currently)
- **Types**: No explicit type hints, use numpy for numeric arrays
- **Documentation**: Docstrings preceding functions with descriptive comments
- **Error Handling**: Simple error handling with parameter validation

## Project Structure
- Primary file: `mobius.py` containing all visualization code
- Visualization shows 3D torus with polygonal cross-section and variable twist
- Interactive UI with sliders for polygon sides and twist parameters

## Development Notes
- Project uses matplotlib for 3D visualization and UI components
- "vibe coding" approach as documented in vibe_coding.md