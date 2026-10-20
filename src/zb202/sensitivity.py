"""Full-factorial assumption stress check using the same authorized policy envelope."""
import copy
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from uuid import uuid4
from .simulation import load, sha256, simulate


def run(root, config_path):
    root = Path(root)
    protocol = load(Path(config_path))
    paths = {'protocol': Path(config_path), 'room': root/'configs/room.json',
             'ahu': root/'configs/ahu.json', 'scenario': root/protocol['base_scenario']}
    room, ahu, scenario = (load(paths[k]) for k in ('room', 'ahu', 'scenario'))
    room_keys = ['volume_m3', 'co2_generation_m3_s_person']
    ahu_keys = ['zone_supply_airflow_m3_s', 'damper_time_constant_s', 'damper_curve_exponent']
    keys = room_keys+ahu_keys
    records = []
    total_supply = ahu['snapshot_units']['supply_airflow_l_s_to_m3_s']
    threshold = protocol['evaluation_target_ppm']
    for index, values in enumerate(product(*(protocol[k] for k in keys))):
        params = dict(zip(keys, values))
        case_room, case_ahu = copy.deepcopy(room), copy.deepcopy(ahu)
        for key in room_keys:
            case_room['parameters'][key]['value'] = params[key]
        for key in ahu_keys:
            case_ahu['parameters'][key]['value'] = params[key]
        for policy in scenario['strategies']:
            rows = simulate(case_room, case_ahu, scenario, policy)
            minutes = [(r['interval_end_s']-r['interval_start_s'])/60 for r in rows]
            records.append(dict(source='synthetic', case=index, **params,
                                room_allocation_fraction=params['zone_supply_airflow_m3_s']/total_supply,
                                strategy=policy['id'],
                                peak_co2_ppm=max([rows[0]['co2_start_ppm']]+[r['co2_end_ppm'] for r in rows]),
                                final_co2_ppm=rows[-1]['co2_end_ppm'],
                                # End-of-minute samples, not exact continuous exceedance time.
                                end_sample_minutes_above_target=sum(m for r, m in zip(rows, minutes) if r['co2_end_ppm'] > threshold),
                                ppm_min_above_target=sum(max(r['co2_end_ppm']-threshold, 0)*m for r, m in zip(rows, minutes)),
                                outdoor_air_volume_m3=sum(r['outdoor_airflow_mean_m3_s']*m*60 for r, m in zip(rows, minutes)),
                                command_min=min(r['damper_command_pct'] for r in rows),
                                command_max=max(r['damper_command_pct'] for r in rows)))
    output = root/'data/synthetic'/(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')+'-sensitivity-'+uuid4().hex[:8])
    output.mkdir(parents=True)
    with (output/'cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = []
    for policy in scenario['strategies']:
        rows = [r for r in records if r['strategy'] == policy['id']]
        summary.append(dict(strategy=policy['id'], cases=len(rows),
                            cases_with_end_sample_exceedance=sum(r['end_sample_minutes_above_target'] > 0 for r in rows),
                            peak_co2_min=min(r['peak_co2_ppm'] for r in rows),
                            peak_co2_max=max(r['peak_co2_ppm'] for r in rows),
                            outdoor_volume_min=min(r['outdoor_air_volume_m3'] for r in rows),
                            outdoor_volume_max=max(r['outdoor_air_volume_m3'] for r in rows)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    manifest = dict(source='synthetic', calibrated=False, protocol=protocol,
                    base_configs={'room': room, 'ahu': ahu, 'scenario': scenario},
                    cases=len(records)//len(scenario['strategies']), records=len(records),
                    inputs={k: {'path': str(v.relative_to(root)), 'sha256': sha256(v)} for k, v in paths.items()},
                    code_sha256={str(p.relative_to(root)): sha256(p) for p in sorted((root/'src/zb202').rglob('*.py'))},
                    outputs={name: sha256(output/name) for name in ['cases.csv', 'summary.json']},
                    limitations=['Stress grid has no probability interpretation or confirmed coverage.',
                                 'All outcomes inherit monotone assumed flow response; actuator command/feedback mapping uncalibrated.',
                                 'No measured room allocation, outdoor fraction, occupancy or volume.',
                                 'End-of-minute exceedance is discretized; no energy or real BA performance claims.'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output, summary
