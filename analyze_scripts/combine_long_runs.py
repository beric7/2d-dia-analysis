from utils.load_json import load_json_data
import pandas as pd
from plotting_functions import get_curve, get_gsd, get_percent_passing
# 'south_high_part_1_144fps' 'vaBeach_part_1_144fps' 'venice_all_long1_144fps' 'venice_part_1_144fps' 'washington_part_1_144fps'

save_dir = '/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/'
sand_name = 'Washington_State'
experiment_name = 'washington_combined_144fps'
percent_values = [0.01, 0.05, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9, 0.95, 0.99]

d1 = load_json_data(f'/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/washington_part_1_144fps/coco_formatted.json')
d2 = load_json_data(f'/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/washington_part_2_144fps/coco_formatted.json')
d3 = load_json_data(f'/projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/washington_part_3_144fps/coco_formatted.json')

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.DataFrame(d3)


df = pd.concat([df1, df2, df3])
df_sorted = df.sort_values(by='volume', ascending=False)
df = df_sorted.iloc[100:]

cs = get_gsd(df, key='ellip_volume')
gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)
# save GSD curve
df_gsd = get_curve(df, save_dir, key='ellip_volume')
df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

comparison_df = pd.read_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv')
comparison_df[experiment_name] = gsd_results['Size']
comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)
