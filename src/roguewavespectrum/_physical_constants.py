"""
Physical constants used in the wave spectrum object.
"""

import linearwavetheory.settings as lwt

# Physical constants
GRAVITATIONAL_ACCELERATION = 9.80665
WATER_DENSITY = 1024.0
WATER_TEMPERATURE_DEGREES_C = 15.0
KINEMATIC_VISCOSITY_WATER = 1.19e-6
VONKARMAN_CONSTANT = 0.4
DYNAMIC_SURFACE_TENSION_WATER = 0.073
PHILLIPS_CONSTANT = 0.0081


class PhysicsOptions:
    """
    Physics options used for calculations in the wave spectrum object. Contains the following attributes:

    - density: density of the fluid
    - temperature: temperature of the fluid
    - dynamic_viscosity: dynamic viscosity of the fluid
    - vonkarman_constant: von Karman constant
    - surface_tension: surface tension of the fluid
    - gravity: gravitational acceleration
    - wave_type: "gravity", "capillary", or "gravity-capillary", determines which dispersion relation to use for
        properties derived from the linear dispersion relation (wavenumber, groupspeed, etc.)
    - wave_regime: "deep", "intermediate", or "shallow", determines which limit of the dispersion relation to use for
        properties derived from the linear dispersion relation (wavenumber, groupspeed, etc.)
    """

    def __init__(
        self,
        density: float = WATER_DENSITY,
        temperature_degrees: float = WATER_TEMPERATURE_DEGREES_C,
        dynamic_viscosity: float = KINEMATIC_VISCOSITY_WATER * WATER_DENSITY,
        vonkarman_constant: float = VONKARMAN_CONSTANT,
        dynamic_surface_tension: float = DYNAMIC_SURFACE_TENSION_WATER * WATER_DENSITY,
        gravity: float = GRAVITATIONAL_ACCELERATION,
        wave_type: str = "gravity",
        wave_regime: str = "intermediate",
    ):
        """
        :param density: Density of the fluid. Default is 1024 kg/m^3
        :param temperature_degrees: temperature of the fluid, Default is 15 degrees C.
        :param dynamic_viscosity: dynamic viscosity of the fluid. Default is 0.00121856 in Pascal second [Pa*s]
        :param vonkarman_constant: von Karman constant
        :param dynamic_surface_tension: dynamic surface tension (N/m). Default is 0.073 N/m
        :param gravity: gravitational acceleration (m^2/s)
        :param wave_type: one of "gravity", "capillary", or "gravity-capillary". Default is "gravity".
        :param wave_regime: one of "deep", "intermediate", or "shallow". Default is "intermediate".
        """
        self.density = density
        self.temperature = temperature_degrees
        self.dynamic_viscosity = dynamic_viscosity
        self.vonkarman_constant = vonkarman_constant
        self.surface_tension = dynamic_surface_tension
        self.gravity = gravity
        self.wave_type = wave_type
        self.wave_regime = wave_regime

    @property
    def kinematic_surface_tension(self):
        return self.surface_tension / self.density

    @property
    def kinematic_viscosity(self):
        return self.dynamic_viscosity / self.density


PHYSICSOPTIONS = PhysicsOptions(
    density=WATER_DENSITY,
    temperature_degrees=WATER_TEMPERATURE_DEGREES_C,
    dynamic_viscosity=KINEMATIC_VISCOSITY_WATER * WATER_DENSITY,
    vonkarman_constant=VONKARMAN_CONSTANT,
    dynamic_surface_tension=DYNAMIC_SURFACE_TENSION_WATER,
    gravity=GRAVITATIONAL_ACCELERATION,
    wave_type="gravity",
    wave_regime="intermediate",
)


def _as_physicsoptions_lwt(physicsoptions: PhysicsOptions) -> lwt.PhysicsOptions:
    """
    Convert a PhysicsOptions object to a PhysicsOptionsLWT object. This is used to pass the PhysicsOptions object
    to the linearwavetheory package.

    :param physicsoptions: PhysicsOptions object
    :return: PhysicsOptionsLWT object
    """
    return lwt.PhysicsOptions(
        kinematic_surface_tension=physicsoptions.kinematic_surface_tension,
        grav=physicsoptions.gravity,
        wave_type=physicsoptions.wave_type,
        wave_regime=physicsoptions.wave_regime,
    )
