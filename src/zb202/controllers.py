"""Baseline policies returning simulated commands only; no BA connectivity."""


def command(policy, co2_ppm):
    if policy['type'] == 'fixed':
        value = policy['command_pct']
    elif policy['type'] == 'proportional':
        low, high = policy['minimum_pct'], policy['maximum_pct']
        if not 0 <= low <= high <= 100 or policy['gain_pct_per_ppm'] < 0:
            raise ValueError('Invalid feedback policy')
        value = min(high, max(low, low + policy['gain_pct_per_ppm'] * (co2_ppm - policy['target_ppm'])))
    else:
        raise ValueError('Unknown policy type: ' + policy['type'])
    if not 0 <= value <= 100:
        raise ValueError('Command must lie between 0 and 100 percent')
    return value
