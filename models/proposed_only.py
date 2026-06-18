import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     Concatenate, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
np.random.seed(42)

FEATURE_COLS = ['Vdc','e_a','e_b','e_c','i_a','i_b','i_c',
                'i_a_ref','i_b_ref','i_c_ref',
                'err_a','err_b','err_c',
                'i_a_prev','i_b_prev','i_c_prev',
                'P','Q','freq_est','THD_est']
N_CLASSES   = 6
CLASS_NAMES = ['Healthy','Open-Circuit','DC Injection',
               'Sag/Swell','Islanding','PLL Desync']
CB = [EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
      ReduceLROnPlateau(patience=8, factor=0.5, verbose=0, monitor='val_accuracy')]

print("Loading dataset...")
df = pd.read_csv('/home/claude/dataset/inverter_3phase_fault_dataset.csv')

# ── SPLIT ─────────────────────────────────────────────────────────────────────
wdf = df.groupby('window_id')['fault_label'].first().reset_index()
tr_ids, tmp = train_test_split(wdf['window_id'], test_size=0.30,
                stratify=wdf['fault_label'], random_state=42)
vl_ids, ts_ids = train_test_split(tmp, test_size=0.50,
                stratify=wdf.loc[wdf['window_id'].isin(tmp),'fault_label'],
                random_state=42)
tr = df[df['window_id'].isin(tr_ids)]
vl = df[df['window_id'].isin(vl_ids)]
ts = df[df['window_id'].isin(ts_ids)]

# ── RICH FEATURES ─────────────────────────────────────────────────────────────
def to_rich(data):
    X, y = [], []
    for wid, g in data.groupby('window_id'):
        f = g[FEATURE_COLS].values.astype(float)
        rms  = np.sqrt((f**2).mean(axis=0))
        skew = ((f-f.mean(0))**3).mean(0) / (f.std(0)+1e-8)**3
        row  = np.concatenate([f.mean(0), f.std(0), f.max(0),
                                f.min(0), f.max(0)-f.min(0), rms, skew])
        y.append(g['fault_label'].iloc[0])
        X.append(row)
    return np.array(X), np.array(y)

print("Building features...")
Xtr, ytr = to_rich(tr)
Xvl, yvl = to_rich(vl)
Xts, yts = to_rich(ts)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xvl = sc.transform(Xvl)
Xts = sc.transform(Xts)
n_feat = Xtr.shape[1]
print(f"Feature dim: {n_feat}, Train: {len(Xtr)}, Val: {len(Xvl)}, Test: {len(Xts)}")

results = {}

def evaluate(name, model, Xt, yt, is_keras=False, t0=0):
    pred = np.argmax(model.predict(Xt, verbose=0), axis=1) if is_keras else model.predict(Xt)
    acc  = accuracy_score(yt, pred)*100
    f1   = f1_score(yt, pred, average='weighted')
    elapsed = time.time()-t0
    results[name] = {'acc':acc,'f1':f1,'pred':pred,'true':yt,'time':elapsed}
    print(f"  {name:<40} Acc={acc:.2f}%  F1={f1:.4f}  ({elapsed:.1f}s)")

# ── BASELINES ─────────────────────────────────────────────────────────────────
print("\n[1/6] Random Forest")
t0=time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr); evaluate("Random Forest", rf, Xts, yts, t0=t0)

print("\n[2/6] XGBoost")
t0=time.time()
xgbm = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='mlogloss', random_state=42, verbosity=0)
xgbm.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
evaluate("XGBoost", xgbm, Xts, yts, t0=t0)

print("\n[3/6] Shallow MLP (Paper 1 baseline)")
t0=time.time()
m3 = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(n_feat,)),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(32,  activation='relu'),
    Dense(N_CLASSES, activation='softmax')
])
m3.compile('adam','categorical_crossentropy',metrics=['accuracy'])
m3.fit(Xtr, to_categorical(ytr,N_CLASSES),
       validation_data=(Xvl, to_categorical(yvl,N_CLASSES)),
       epochs=300, batch_size=64, callbacks=CB, verbose=0)
evaluate("Shallow MLP", m3, Xts, yts, is_keras=True, t0=t0)

print("\n[4/6] Deep MLP")
t0=time.time()
inp4 = Input(shape=(n_feat,))
x4   = Dense(256, activation='relu')(inp4)
x4   = BatchNormalization()(x4)
x4   = Dropout(0.3)(x4)
x4   = Dense(256, activation='relu')(x4)
x4   = BatchNormalization()(x4)
x4   = Dropout(0.3)(x4)
x4   = Dense(128, activation='relu')(x4)
x4   = BatchNormalization()(x4)
x4   = Dropout(0.25)(x4)
x4   = Dense(64,  activation='relu')(x4)
out4 = Dense(N_CLASSES, activation='softmax')(x4)
m4   = Model(inp4, out4)
m4.compile(tf.keras.optimizers.Adam(5e-4),'categorical_crossentropy',metrics=['accuracy'])
m4.fit(Xtr, to_categorical(ytr,N_CLASSES),
       validation_data=(Xvl, to_categorical(yvl,N_CLASSES)),
       epochs=300, batch_size=64, callbacks=CB, verbose=0)
evaluate("Deep MLP", m4, Xts, yts, is_keras=True, t0=t0)

print("\n[5/6] Gradient Boosting")
t0=time.time()
gb = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                 learning_rate=0.1, random_state=42)
gb.fit(Xtr, ytr); evaluate("Gradient Boosting", gb, Xts, yts, t0=t0)

# ── PROPOSED: Residual MLP (ResNet-style for tabular data) ───────────────────
# Clean design:
#   • Residual blocks: skip connections prevent vanishing gradient
#   • GELU activations: smoother than ReLU, better for fault boundary detection
#   • BatchNorm after each block
#   • Wider first layer to capture cross-feature interactions
#   • No attention/gate complexity — stability over cleverness
print("\n[6/6] Proposed Residual MLP (ResMLP)")
t0=time.time()

def res_block(x, units):
    skip = Dense(units)(x)          # project skip to same dim
    x    = Dense(units, activation='gelu')(x)
    x    = BatchNormalization()(x)
    x    = Dropout(0.2)(x)
    x    = Dense(units, activation='gelu')(x)
    x    = BatchNormalization()(x)
    x    = Add()([x, skip])         # residual connection
    return x

inp6 = Input(shape=(n_feat,))
x6   = Dense(256, activation='gelu')(inp6)
x6   = BatchNormalization()(x6)
x6   = res_block(x6, 256)
x6   = res_block(x6, 256)
x6   = Dropout(0.25)(x6)
x6   = res_block(x6, 128)
x6   = res_block(x6, 128)
x6   = Dropout(0.2)(x6)
x6   = Dense(64, activation='gelu')(x6)
x6   = BatchNormalization()(x6)
x6   = Dropout(0.15)(x6)
out6 = Dense(N_CLASSES, activation='softmax')(x6)

proposed = Model(inp6, out6, name='ResMLP')
proposed.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy', metrics=['accuracy'])
proposed.summary()

history = proposed.fit(
    Xtr, to_categorical(ytr, N_CLASSES),
    validation_data=(Xvl, to_categorical(yvl, N_CLASSES)),
    epochs=400, batch_size=64, callbacks=CB, verbose=1)
evaluate("Proposed ResMLP", proposed, Xts, yts, is_keras=True, t0=t0)

# ── FINAL TABLE ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Model':<40} {'Accuracy':>10} {'F1-Score':>10} {'Time':>7}")
print("="*70)
for m_name, r in results.items():
    marker = " ◄ PROPOSED" if "ResMLP" in m_name else ""
    print(f"{m_name:<40} {r['acc']:>9.2f}% {r['f1']:>10.4f} {r['time']:>6.1f}s{marker}")
print("="*70)

pd.DataFrame([{'Model':k,'Accuracy(%)':f"{v['acc']:.2f}",
               'F1-Score':f"{v['f1']:.4f}"} for k,v in results.items()]
             ).to_csv('/home/claude/dataset/results_final.csv', index=False)

# ── ALL FIGURES ───────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({'font.size':11,'font.family':'serif'})
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
f1s    = [results[n]['f1']  for n in names]
colors = ['#4C72B0','#4C72B0','#4C72B0','#4C72B0','#4C72B0','#DD4444']

# Fig 1
fig, ax = plt.subplots(figsize=(11,5))
bars = ax.bar(names, accs, color=colors, edgecolor='black', width=0.55)
ax.set_ylim(min(accs)*0.95, 102)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax.set_title('Fig. 1 — Model Accuracy Comparison\nThree-Phase Grid-Tied Inverter Fault Detection', fontsize=12)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.axhline(max(accs), color='red', linestyle='--', alpha=0.4, label=f'Best: {max(accs):.2f}%')
ax.legend(fontsize=10)
plt.xticks(rotation=20, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig1_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 2 — confusion matrix
cm = confusion_matrix(results['Proposed ResMLP']['true'],
                       results['Proposed ResMLP']['pred'])
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, annot_kws={'size':11})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Fig. 2 — Confusion Matrix: Proposed ResMLP\nThree-Phase Inverter Fault Classification', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig2_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 3 — per-class F1
fig, ax = plt.subplots(figsize=(12,5))
x     = np.arange(N_CLASSES)
width = 0.13
clr   = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974','#DD4444']
for i,(mn,r) in enumerate(results.items()):
    rpt = classification_report(r['true'],r['pred'],output_dict=True,zero_division=0)
    f1s_c = [rpt[str(c)]['f1-score'] for c in range(N_CLASSES)]
    ax.bar(x+i*width, f1s_c, width, label=mn, color=clr[i],
           edgecolor='black', linewidth=0.5)
ax.set_xticks(x+width*2.5)
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title('Fig. 3 — Per-Class F1-Score Across All Models', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig3_f1_perclass.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 4 — training history
fig, axes = plt.subplots(1,2, figsize=(11,4))
axes[0].plot(history.history['accuracy'],    label='Train', color='#4C72B0')
axes[0].plot(history.history['val_accuracy'],label='Val',   color='#DD4444')
axes[0].set_title('Training Accuracy — Proposed ResMLP')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'],    label='Train', color='#4C72B0')
axes[1].plot(history.history['val_loss'],label='Val',   color='#DD4444')
axes[1].set_title('Training Loss — Proposed ResMLP')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('Fig. 4 — Proposed ResMLP Training Curves', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig4_training.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 5 — waveform comparison
raw = pd.read_csv('/home/claude/dataset/inverter_3phase_fault_dataset.csv')
WINDOW_SIZE = 50
fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)
t_ms = np.arange(WINDOW_SIZE)/5000*1000
h_wid = raw[raw['fault_label']==0]['window_id'].iloc[0]
f_wid = raw[raw['fault_label']==1]['window_id'].iloc[0]
h_data = raw[raw['window_id']==h_wid]
f_data = raw[raw['window_id']==f_wid]
for ph,col,c in zip(['A','B','C'],['i_a','i_b','i_c'],
                     ['#1f77b4','#ff7f0e','#2ca02c']):
    axes[0].plot(t_ms, h_data[col].values, color=c, label=f'Phase {ph}', lw=1.5)
    axes[1].plot(t_ms, f_data[col].values, color=c, label=f'Phase {ph}',
                 lw=1.5, linestyle='--' if ph=='A' else '-')
axes[0].set_title('(a) Healthy — Balanced Three-Phase Currents', fontsize=11)
axes[1].set_title('(b) Open-Circuit Fault — Phase A Half-Cycle Loss', fontsize=11)
for ax in axes:
    ax.set_ylabel('Current (A)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
axes[1].set_xlabel('Time (ms)', fontsize=10)
plt.suptitle('Fig. 5 — Three-Phase Current Waveforms: Healthy vs Fault', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig5_waveforms.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 6 — THD per class
fig, ax = plt.subplots(figsize=(8,5))
thd = raw.groupby('fault_name')['THD_est'].mean().sort_values()
clrs_t = ['#2ca02c' if 'Healthy' in n else '#4C72B0' for n in thd.index]
bars_t = ax.barh(thd.index, thd.values, color=clrs_t, edgecolor='black')
ax.set_xlabel('Mean THD Estimate (%)', fontsize=12)
ax.set_title('Fig. 6 — Mean THD by Fault Class', fontsize=12)
for bar,val in zip(bars_t, thd.values):
    ax.text(val+0.1, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/dataset/fig6_thd.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All 6 figures saved.")
print("\nClassification report — Proposed ResMLP:")
print(classification_report(results['Proposed ResMLP']['true'],
                             results['Proposed ResMLP']['pred'],
                             target_names=CLASS_NAMES))
