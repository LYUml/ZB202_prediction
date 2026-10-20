"""Scenario runner: config -> model -> explicitly synthetic CSV and provenance."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from . import __version__
from .adapters.scenario import ScenarioInputs
from .controllers import command
from .models.ahu import actuator, outdoor_airflow
from .models.co2 import advance

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    data = json.loads(path.read_text())
    def check(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Configuration values must be finite')
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        if isinstance(value, list):
            for child in value:
                check(child)
    check(data)
    return data


def simulate(room, ahu, scenario, policy):
    r = {k: v['value'] for k, v in room['parameters'].items()}
    a = {k: v['value'] for k, v in ahu['parameters'].items()}
    envelope = ahu.get('simulation_envelope', {}).get('damper_command_pct', {})
    minimum, maximum = envelope.get('minimum', 0), envelope.get('maximum', 100)
    if not 0 <= minimum <= maximum <= 100:
        raise ValueError('Invalid simulated damper operating envelope')
    if policy['type'] == 'fixed':
        if not minimum <= policy['command_pct'] <= maximum:
            raise ValueError('Fixed policy exceeds the authorized simulation envelope')
    elif policy['type'] == 'proportional':
        if policy['minimum_pct'] < minimum or policy['maximum_pct'] > maximum:
            raise ValueError('Feedback policy exceeds the authorized simulation envelope')
    dt = scenario['step_seconds']
    duration = scenario['duration_minutes'] * 60
    if dt <= 0 or duration <= 0 or duration % dt != 0:
        raise ValueError('Duration must be positive and divisible by time step')
    inputs = ScenarioInputs(scenario['schedule'])
    if any(item['minute'] * 60 % dt != 0 for item in scenario['schedule']):
        raise ValueError('Occupancy changes must align with simulation steps')
    if r['volume_m3'] <= 0 or min(r['infiltration_ach'], r['outdoor_co2_ppm'], r['co2_generation_m3_s_person']) < 0:
        raise ValueError('Invalid room parameters')
    co2 = scenario['initial_co2_ppm']
    position = scenario['initial_damper_position_pct']
    if co2 < 0:
        raise ValueError('Initial CO2 must be nonnegative')
    infiltration = r['infiltration_ach'] * r['volume_m3'] / 3600
    rows = []
    elapsed = 0
    while elapsed < duration:
        occupants = inputs.occupants_at(elapsed)
        control = command(policy, co2)
        if not minimum <= control <= maximum:
            raise ValueError('Simulated command exceeds the authorized operating envelope')
        initial_co2, initial_position = co2, position
        remaining = dt
        flow_integral = 0
        # Resolve actuator lag with at most 5-second midpoint flow substeps.
        while remaining > 0:
            step = min(5, remaining)
            midpoint = actuator(position, control, a['damper_time_constant_s'], step / 2)
            flow = outdoor_airflow(midpoint, a['zone_supply_airflow_m3_s'], a['outdoor_fraction_closed'], a['outdoor_fraction_open'], a['damper_curve_exponent'])
            co2 = advance(co2, r['outdoor_co2_ppm'], r['volume_m3'], occupants, r['co2_generation_m3_s_person'], flow + infiltration, step)
            position = actuator(position, control, a['damper_time_constant_s'], step)
            flow_integral += flow * step
            remaining -= step
        rows.append(dict(source='synthetic', strategy=policy['id'], interval_start_s=elapsed,
                         interval_end_s=elapsed + dt, occupants_assumed=occupants,
                         co2_start_ppm=initial_co2, co2_end_ppm=co2,
                         damper_command_pct=control, damper_position_start_pct=initial_position,
                         damper_position_end_pct=position, outdoor_airflow_mean_m3_s=flow_integral / dt,
                         infiltration_m3_s=infiltration,
                         fan_command_baseline_pct=ahu.get('snapshot_only', {}).get('fan_command_pct')))
        elapsed += dt
    return rows


def run(project_root, scenario_path):
    paths = {'room': project_root / 'configs/room.json', 'ahu': project_root / 'configs/ahu.json', 'scenario': scenario_path}
    configs = {key: load(path) for key, path in paths.items()}
    policies = configs['scenario']['strategies']
    ids = [p['id'] for p in policies]
    if not policies or len(ids) != len(set(ids)):
        raise ValueError('Strategies must be nonempty with unique IDs')
    # Calculate and validate before creating output files.
    results = [simulate(configs['room'], configs['ahu'], configs['scenario'], p) for p in policies]
    now = datetime.now(timezone.utc)
    name = now.strftime('%Y-%m-%dT%H-%M-%S') + '-' + uuid4().hex[:8]
    output = project_root / 'data/synthetic' / name
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'timeseries.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0][0]))
        writer.writeheader()
        for rows in results:
            writer.writerows(rows)
    summary = [{
        'strategy': rows[0]['strategy'],
        'peak_co2_ppm': max([rows[0]['co2_start_ppm']] + [r['co2_end_ppm'] for r in rows]),
        'final_co2_ppm': rows[-1]['co2_end_ppm'],
        'outdoor_air_volume_m3': sum(r['outdoor_airflow_mean_m3_s'] * (r['interval_end_s'] - r['interval_start_s']) for r in rows),
    } for rows in results]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = {
        'source': 'synthetic', 'calibrated': False, 'version': __version__,
        'created_at_utc': now.isoformat(), 'configs': configs,
        'inputs': {key: {'path': str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path), 'sha256': sha256(path)} for key, path in paths.items()},
        'code_sha256': {str(p.relative_to(project_root)): sha256(p) for p in sorted((project_root / 'src/zb202').rglob('*.py'))},
        'outputs': {p.name: {'sha256': sha256(p)} for p in [csv_path, output / 'summary.json']},
        'rows': sum(len(rows) for rows in results),
        'limitations': ['Illustrative assumptions only; not a validated ZB202 response model.',
                        'No BA connection, no historical sensor replay, no real actuator commands.',
                        'BA baseline is user-confirmed; fixed room allocation and outdoor-fraction response remain uncalibrated.',
                        'Single well-mixed zone, no thermal/humidity/PM/energy model; air volume is not energy savings.',
                        'Each CSV row describes one interval; occupancy and command apply throughout that interval.'],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output, summary


def main():
    parser = argparse.ArgumentParser(description='Run assumed ZB202 CO2 scenarios; writes only synthetic results.')
    parser.add_argument('--scenario', default='configs/scenarios/demo.json', help='Path relative to project root, or absolute')
    args = parser.parse_args()
    output, summary = run(PROJECT_ROOT, (PROJECT_ROOT / args.scenario).resolve())
    print('Synthetic output:', output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
