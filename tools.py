def calculate_reynolds_number(density, velocity, diameter, viscosity):
    """
    Calculate Reynolds number.

    Re = (rho * v * D) / mu
    """

    reynolds_number = (
        density * velocity * diameter
    ) / viscosity

    return reynolds_number


def calculate_ideal_gas_volume(pressure, moles, temperature, gas_constant=8.314):
    """
    Calculate volume using the ideal gas law.

    PV = nRT
    V = nRT / P
    """

    volume = (moles * gas_constant * temperature) / pressure

    return volume


def calculate_heat_duty(mass_flow_rate, specific_heat, temperature_change):
    """
    Calculate sensible heat duty.

    Q = m * Cp * Delta T
    """

    heat_duty = (
        mass_flow_rate
        * specific_heat
        * temperature_change
    )

    return heat_duty
if __name__ == "__main__":

    re = calculate_reynolds_number(
        1000,
        2,
        0.05,
        0.001
    )

    print("Reynolds Number:", re)

    volume = calculate_ideal_gas_volume(
        101325,
        1,
        300
    )

    print("Ideal Gas Volume:", volume)

    heat = calculate_heat_duty(
        2,
        4.18,
        20
    )

    print("Heat Duty:", heat)