"""Assumed actuator and room outdoor-air fraction; not a calibrated AHU."""
import math


def actuator(position_pct, command_pct, tau_s, dt_s):
    if not 0 <= position_pct <= 100 or not 0 <= command_pct <= 100 or tau_s <= 0 or dt_s <= 0:
        raise ValueError('Invalid damper position, command, time constant or step')
    return command_pct + (position_pct - command_pct) * math.exp(-dt_s / tau_s)


def outdoor_airflow(position_pct, supply_m3_s, closed_fraction, open_fraction, exponent):
    if not 0 <= position_pct <= 100 or supply_m3_s < 0 or not 0 <= closed_fraction <= open_fraction <= 1 or exponent <= 0:
        raise ValueError('Invalid outdoor airflow parameters')
    fraction = closed_fraction + (open_fraction - closed_fraction) * (position_pct / 100) ** exponent
    return supply_m3_s * fraction
