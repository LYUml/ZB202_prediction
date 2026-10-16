"""Well-mixed single-zone mass balance; CO2 in ppm, flows in m3/s."""
import math


def advance(co2_ppm, outdoor_ppm, volume_m3, occupants, emission_m3_s_person,
            outdoor_airflow_m3_s, dt_s):
    """Exact solution for constant source and flow over one integration step."""
    if volume_m3 <= 0 or dt_s <= 0 or min(occupants, emission_m3_s_person, outdoor_airflow_m3_s) < 0:
        raise ValueError('Invalid room volume, time step, occupancy, emission or airflow')
    source = occupants * emission_m3_s_person * 1e6 / volume_m3
    rate = outdoor_airflow_m3_s / volume_m3
    if rate == 0:
        return co2_ppm + source * dt_s
    decay = math.exp(-rate * dt_s)
    return outdoor_ppm + (co2_ppm - outdoor_ppm) * decay + source * (-math.expm1(-rate * dt_s)) / rate
