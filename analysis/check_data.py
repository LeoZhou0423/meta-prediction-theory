import pandas as pd
import numpy as np
from pathlib import Path

# Project root: parent of analysis/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load data
bdi = pd.read_csv(PROJECT_ROOT / 'data' / 'nimh_dataset' / 'ds005752' / 'phenotype' / 'bdi.tsv', sep='\t')
phq9 = pd.read_csv(PROJECT_ROOT / 'data' / 'nimh_dataset' / 'ds005752' / 'phenotype' / 'phq9.tsv', sep='\t')

# Calculate BDI total score
bdi_items = [col for col in bdi.columns if col.startswith('Q') and col[1:].isdigit()]
bdi['bdi_total'] = bdi[bdi_items].replace(-999, np.nan).sum(axis=1)

# Calculate PHQ-9 total score
phq9_items = ['LITTLE_INTEREST', 'FEELING_DOWN', 'TROUBLE_SLEEPING', 'FEELING_TIRED', 
              'POOR_APPETITE', 'FEELING_BAD', 'TROUBLE_CONCENTRATING', 'MOVING_OR_SPEAKING', 
              'BETTER_OFF_DEAD']
phq9['phq9_total'] = phq9[phq9_items].sum(axis=1)

# Find common participants
common_ids = set(bdi['participant_id']) & set(phq9['participant_id'])
print(f"Total common participants: {len(common_ids)}")

# Check visits for common participants
print("\nCommon participant visits:")
for pid in sorted(list(common_ids))[:10]:
    bdi_visits = bdi[bdi['participant_id'] == pid]['visit'].tolist()
    phq9_visits = phq9[phq9['participant_id'] == pid]['visit'].tolist()
    print(f"{pid}: BDI={bdi_visits}, PHQ-9={phq9_visits}")

# Try different merge strategies
print("\nMerge strategies:")

# Strategy 1: Merge on participant_id only (ignoring visit)
merged1 = pd.merge(
    bdi[['participant_id', 'age_at_visit', 'bdi_total']],
    phq9[['participant_id', 'age_at_visit', 'phq9_total']],
    on='participant_id',
    how='inner'
)
print(f"Strategy 1 (merge on participant_id only): {len(merged1)} matches")

# Strategy 2: Use first visit for each participant
bdi_first = bdi.sort_values('visit').drop_duplicates('participant_id', keep='first')
phq9_first = phq9.sort_values('visit').drop_duplicates('participant_id', keep='first')
merged2 = pd.merge(
    bdi_first[['participant_id', 'age_at_visit', 'bdi_total']],
    phq9_first[['participant_id', 'age_at_visit', 'phq9_total']],
    on='participant_id',
    how='inner'
)
print(f"Strategy 2 (first visit): {len(merged2)} matches")

# Strategy 3: Use all visits (cross-join)
merged3 = pd.merge(
    bdi[['participant_id', 'visit', 'age_at_visit', 'bdi_total']],
    phq9[['participant_id', 'visit', 'age_at_visit', 'phq9_total']],
    on=['participant_id', 'visit'],
    how='inner'
)
print(f"Strategy 3 (same visit): {len(merged3)} matches")
