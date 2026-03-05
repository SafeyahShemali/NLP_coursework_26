import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F7F7F7',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
})

COLOR_NEG = '#3E78B5'   # for No PCL
COLOR_POS = '#BA2414'   # for PCL

#1- Loading data
data_path = "dontpatronizeme_pcl.tsv"

df = pd.read_csv(data_path,sep='\t', header=None, names=['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label'], quoting=3,)
df = df.dropna(subset=['text'])
df['label'] = df['label'].astype(int)

#2- Binary Label Creation based on the paper (Section 4.2.1):
#   Labels 0, 1 → Negative (No PCL)
#   Labels 2, 3, 4 → Positive (PCL)

# convert (0 - 5) scale to  (0-1) scale
df['binary_label'] = df['label'].apply(lambda x: 0 if (x== 0 or x==1) else 1)
# textual labeling of the paragraphs 
df['class_name'] = df['binary_label'].map({0: 'No PCL', 1: 'PCL'})

# based on the paper, the expected is ~995 positive, ~9642 negative
print(f"Positive (PCL):   {df['binary_label'].sum()}")
print(f"Negative (No PCL): {(df['binary_label'] == 0).sum()}")


# EDA tech 1: Class Distribution  ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Figure1-A: 5-Point Label Distribution ──────────────────────────────────────
label_counts = df['label'].value_counts().sort_index()
ax1 = axes[0]
ax1.bar(label_counts.index, label_counts.values, color=['#009BFF', '#6B98B5', '#C9B86B', '#D97921', '#FF4A38'], edgecolor='white')
ax1.set_xlabel('Annotation Label (0-4 scale)')
ax1.set_ylabel('Paragraphs Count')
ax1.set_title('(A) Fine-Grained Label Distribution')


# Figure1-B: Binary Class Split ──────────────────────────────────────────────
binary_counts = df['class_name'].value_counts()
ax2 = axes[1]
ax2.bar(['No PCL', 'PCL'],binary_counts.values, color =[COLOR_NEG, COLOR_POS], edgecolor='white' )
ax2.set_ylabel('Number of Paragraphs')
ax2.set_title('(B) Binary Classification Split')

# Calculate imbalance ratio
n_pos = df['binary_label'].sum()
n_neg = (df['binary_label'] == 0).sum()
ratio = n_neg / n_pos
ax2.annotate(f'Imbalance Ratio: {ratio:.1f}:1', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='#FDEBD0'))
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# EDA tech 1: Text Length Distribution  ──────────────────────────────────────
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
print("── Text Length Statistics ──")
for cls_name, group in df.groupby('class_name'):
    wc = group['word_count']
    print(f"  {cls_name}: mean={wc.mean():.1f}, median={wc.median():.0f}, "
          f"min={wc.min()}, max={wc.max()}")


# Figure2-A ────────────────────────────────────────────────────────
nopcl_words = df[df['binary_label'] == 0]['word_count']
pcl_words   = df[df['binary_label'] == 1]['word_count']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
bins = np.arange(0, df['word_count'].quantile(0.99) + 10, 5)
ax1.hist(nopcl_words, bins=bins, alpha=0.6, color=COLOR_NEG, label='No PCL', density=True)
ax1.hist(pcl_words,   bins=bins, alpha=0.6, color=COLOR_POS, label='PCL', density=True)
ax1.set_xlabel('Word Count')
ax1.set_ylabel('Density')
ax1.set_title('(A) Word Count Distribution')
ax1.legend()


# Figure2-B: Box Plot
ax2 = axes[1]
bp = ax2.boxplot([nopcl_words.values, pcl_words.values], labels=['No PCL', 'PCL'], patch_artist=True)
bp['boxes'][0].set_facecolor(COLOR_NEG)
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor(COLOR_POS)
bp['boxes'][1].set_alpha(0.6)
ax2.set_ylabel('Word Count')
ax2.set_title('(B) Word Count Comparison')
ax2.axhline(394, color='orange', linestyle=':', linewidth=2)
ax2.text(2.3, 400, '≈512 tokens', fontsize=9, color='orange', fontweight='bold')
plt.tight_layout()
plt.savefig('text_length.png', dpi=150, bbox_inches='tight')
plt.show()


print("── GENERAL DETAILS FROM HISTOGRAM ──")
pcl_mean = df[df['binary_label'] == 1]['word_count'].mean()
nopcl_mean = df[df['binary_label'] == 0]['word_count'].mean()
print(f"PCL paragraphs: mean word count = {pcl_mean:.1f}")
print(f"No PCL paragraphs: mean word count = {nopcl_mean:.1f}")

# EDA tech 3: DATA QUALITY  ──────────────────────────────────────
print("── Data Quality Checks ──")
n_dup = df.duplicated(subset=['text']).sum()
print(f"Duplicate texts: {n_dup}")

# Check the shortest test 
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
short = (df['word_count'] < 5).sum()
long = (df['word_count'] > 394).sum()
print(f"Very short paragraphs -> <5 words: {short}")
print(f"Very long paragraphs -> exceeds 512 tokens: {long}")

# Check for HTML artifacts
html_pat = re.compile(r'&amp;|&lt;|&gt;|<[^>]+>|\\n|\\t')
has_html = df['text'].apply(lambda x: bool(html_pat.search(str(x)))).sum()
print(f"Paragraphs with HTML artifacts: {has_html}")

'''
Sample of output:
Positive (PCL):   993
Negative (No PCL): 9475
── Text Length Statistics ──
  No PCL: mean=47.9, median=42, min=1, max=909
  PCL: mean=53.6, median=47, min=6, max=512
── GENERAL DETAILS FROM HISTOGRAM ──
PCL paragraphs: mean word count = 53.6
No PCL paragraphs: mean word count = 47.9
── Data Quality Checks ──
Duplicate texts: 0
Very short paragraphs -> <5 words: 13
Very long paragraphs -> exceeds 512 tokens: 3
Paragraphs with HTML artifacts: 469
'''
