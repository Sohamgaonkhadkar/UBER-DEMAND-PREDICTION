"""Render the frozen fitted centroids; does not fit or alter a model."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]

def main():
    source = ROOT / 'artifacts/advanced_20261102/geography.json'
    geography = json.loads(source.read_text(encoding='utf-8'))
    centers = np.asarray(geography['centers'], dtype=float)
    points = centers * np.asarray(geography['scale']) + np.asarray(geography['mean'])
    assert points.shape == (30, 2) and np.isfinite(points).all()
    lon, lat = points.T
    detail = (-74.02, -73.93, 40.70, 40.835)
    inside = (lon >= detail[0]) & (lon <= detail[1]) & (lat >= detail[2]) & (lat <= detail[3])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.35, .85]})
    for ax in axes:
        ax.set_facecolor('#f5f8fc')
        ax.grid(color='#ccd5e0', alpha=.65, linewidth=.7)
        ax.set_axisbelow(True)
        ax.scatter(lon, lat, s=64, c='#1267a5', edgecolors='white', linewidths=1, zorder=3)
        ax.set_xlabel('Longitude (degrees)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylabel('Latitude (degrees)')
        ax.set_aspect(1 / math.cos(math.radians(float(lat.mean()))))
        for spine in ax.spines.values():
            spine.set_color('#94a3b8')
    axes[0].set_xlim(lon.min()-.018, lon.max()+.018)
    axes[0].set_ylim(lat.min()-.018, lat.max()+.018)
    axes[0].set_title('All 30 fitted NYC centroids', loc='left', fontweight='bold', pad=13)
    axes[0].add_patch(Rectangle((detail[0], detail[2]), detail[1]-detail[0], detail[3]-detail[2],
                                fill=False, edgecolor='#e77332', linewidth=1.4, linestyle='--'))
    axes[0].text(-73.922, 40.813, 'Central detail\nshown at right', color='#b14b13', fontsize=9)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=2))
    axes[1].set_xlim(detail[0], detail[1])
    axes[1].set_ylim(detail[2], detail[3])
    axes[1].set_title('Central NYC detail', loc='left', fontweight='bold', pad=13)
    offsets = {12:(-24,5), 1:(7,-12), 14:(8,-3), 15:(-32,-5), 5:(7,7), 18:(7,-5), 19:(-20,7),
               25:(-32,12), 27:(-26,-9), 7:(7,-1), 26:(7,-7), 10:(-20,-7), 4:(-20,6), 22:(-21,5)}
    for i, (x, y) in enumerate(points):
        ax = axes[1] if inside[i] else axes[0]
        ax.annotate(f'{i:02d}', (x, y), xytext=offsets.get(i, (6,6)), textcoords='offset points',
                    fontsize=8.5, color='#18334f', weight='bold',
                    arrowprops={'arrowstyle':'-', 'color':'#7d8fa4', 'lw':.6}, zorder=4)
    fig.suptitle('UrbanFlow AI - verified geographic model-region centers',
                 x=.065, ha='left', fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=.065, right=.97, bottom=.22, top=.84, wspace=.25)
    fig.text(.065, .045, 'Source: frozen geography.json; inverse StandardScaler transform, [longitude, latitude].\n'
             'Numbers are model region IDs. Dots are centroids; no cluster polygons, demand intensity or basemap is inferred.',
             color='#475569', fontsize=9)
    target = Path(__file__).with_name('verified-region-centroids.png')
    fig.savefig(target, dpi=180, facecolor='white')
    plt.close(fig)
    print(target)

if __name__ == '__main__':
    main()
