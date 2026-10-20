"""
Generates the self-contained KICK interactive dashboard HTML file.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = Path(r"C:\Users\kevin\.gemini\antigravity\brain\578c9837-618f-46cb-882b-dbb9abef28d3")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROCESSED_DIR / "test_shots_web.json", "r", encoding="utf-8") as f:
    shots_data = json.load(f)

with open(PROCESSED_DIR / "xg_model_evaluation_report.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KICK — Football Intelligence Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .pitch-svg {{
      background: radial-gradient(circle at 50% 50%, #1e3a29 0%, #13271b 100%);
      box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }}
    .shot-dot {{
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
    }}
    .shot-dot:hover {{
      transform: scale(1.6);
      filter: drop-shadow(0 0 8px rgba(255,255,255,0.8));
    }}
    .custom-scrollbar::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scrollbar::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.15);
      border-radius: 3px;
    }}
  </style>
</head>
<body class="bg-[#0b0f17] text-[#e2e8f0] font-sans antialiased min-h-screen p-4 md:p-6 custom-scrollbar">

  <!-- Header -->
  <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
    <div class="flex items-center gap-3">
      <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20 font-bold text-white text-lg">
        ⚽
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          KICK <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold uppercase tracking-wider">v1.0 Milestone 2</span>
        </h1>
        <p class="text-xs text-slate-400">Football Intelligence & xG Decision-Support Platform · Premier League 2015/2016 Held-Out Evaluation</p>
      </div>
    </div>
    <div class="flex items-center gap-2 text-xs">
      <span class="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 font-medium">
        🎯 <strong>250</strong> Held-Out Test Shots
      </span>
      <span class="px-3 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800/40 text-emerald-300 font-medium">
        ⚽ <strong>20</strong> Actual Goals
      </span>
      <span class="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-cyan-300 font-medium">
        🛡️ Zero-Leakage Split
      </span>
    </div>
  </header>

  <!-- Metric Overview Cards -->
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 my-6">
    <!-- Card 1: Full Logistic Regression -->
    <div class="bg-slate-900/80 border-2 border-emerald-500/40 rounded-xl p-4 relative overflow-hidden shadow-sm">
      <div class="absolute -right-4 -bottom-4 w-20 h-20 bg-emerald-500/10 rounded-full blur-xl"></div>
      <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span class="font-semibold text-emerald-400">RECOMMENDED MODEL</span>
        <span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[10px]">Full Features</span>
      </div>
      <h3 class="font-bold text-base text-white">Logistic Regression</h3>
      <div class="mt-3 flex items-baseline justify-between">
        <div>
          <span class="text-2xl font-black text-emerald-400">0.840</span>
          <span class="text-xs text-slate-400 ml-1">ROC-AUC</span>
        </div>
        <div class="text-right">
          <span class="text-sm font-semibold text-slate-200">0.2289</span>
          <span class="text-[10px] text-slate-400 block">Log Loss</span>
        </div>
      </div>
      <div class="mt-2 text-xs text-slate-400 flex justify-between border-t border-slate-800/60 pt-2">
        <span>ECE: <strong class="text-slate-200">0.0269</strong></span>
        <span>Pred xG: <strong class="text-emerald-400">23.5</strong> / 20</span>
      </div>
    </div>

    <!-- Card 2: Baseline Model -->
    <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
      <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span>BASELINE BENCHMARK</span>
        <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">Dist + Angle</span>
      </div>
      <h3 class="font-bold text-base text-slate-200">Logistic (Baseline)</h3>
      <div class="mt-3 flex items-baseline justify-between">
        <div>
          <span class="text-2xl font-black text-slate-200">0.769</span>
          <span class="text-xs text-slate-400 ml-1">ROC-AUC</span>
        </div>
        <div class="text-right">
          <span class="text-sm font-semibold text-slate-300">0.2533</span>
          <span class="text-[10px] text-slate-400 block">Log Loss</span>
        </div>
      </div>
      <div class="mt-2 text-xs text-slate-400 flex justify-between border-t border-slate-800/60 pt-2">
        <span>ECE: <strong class="text-slate-200">0.0201</strong></span>
        <span>Pred xG: <strong class="text-slate-200">24.7</strong> / 20</span>
      </div>
    </div>

    <!-- Card 3: Gradient Boosting -->
    <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
      <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span>NON-LINEAR CANDIDATE</span>
        <span class="px-1.5 py-0.5 rounded bg-slate-800 text-purple-300 font-mono text-[10px]">HistGBM</span>
      </div>
      <h3 class="font-bold text-base text-slate-200">Gradient Boosting</h3>
      <div class="mt-3 flex items-baseline justify-between">
        <div>
          <span class="text-2xl font-black text-purple-400">0.773</span>
          <span class="text-xs text-slate-400 ml-1">ROC-AUC</span>
        </div>
        <div class="text-right">
          <span class="text-sm font-semibold text-slate-300">0.2826</span>
          <span class="text-[10px] text-slate-400 block">Log Loss</span>
        </div>
      </div>
      <div class="mt-2 text-xs text-slate-400 flex justify-between border-t border-slate-800/60 pt-2">
        <span>ECE: <strong class="text-slate-200">0.0665</strong></span>
        <span>Pred xG: <strong class="text-purple-400">21.2</strong> / 20</span>
      </div>
    </div>

    <!-- Card 4: StatsBomb Proprietary -->
    <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-4 relative overflow-hidden">
      <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span>INDUSTRY BENCHMARK</span>
        <span class="px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 font-mono text-[10px]">StatsBomb</span>
      </div>
      <h3 class="font-bold text-base text-slate-200">StatsBomb xG</h3>
      <div class="mt-3 flex items-baseline justify-between">
        <div>
          <span class="text-2xl font-black text-cyan-400">0.886</span>
          <span class="text-xs text-slate-400 ml-1">ROC-AUC</span>
        </div>
        <div class="text-right">
          <span class="text-sm font-semibold text-slate-300">0.2027</span>
          <span class="text-[10px] text-slate-400 block">Log Loss</span>
        </div>
      </div>
      <div class="mt-2 text-xs text-slate-400 flex justify-between border-t border-slate-800/60 pt-2">
        <span>ECE: <strong class="text-slate-200">0.0422</strong></span>
        <span>Pred xG: <strong class="text-cyan-400">24.3</strong> / 20</span>
      </div>
    </div>
  </div>

  <!-- Main Grid: Pitch Visualizer & Controls -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
    
    <!-- Left Column: Interactive Pitch (8 cols) -->
    <div class="lg:col-span-8 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h2 class="font-bold text-lg text-white flex items-center gap-2">
            🏟️ Attacking Half Shot Map
          </h2>
          <p class="text-xs text-slate-400">Hover over any shot to inspect probability, distance, and shooter attributes.</p>
        </div>
        
        <!-- Model Selector Toggle -->
        <div class="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          <button id="btn-model-lr-full" onclick="setActiveModel('xg_lr_full')" class="px-3 py-1.5 rounded-lg bg-emerald-500 font-semibold text-white shadow-sm transition">
            Full Logistic
          </button>
          <button id="btn-model-lr-base" onclick="setActiveModel('xg_lr_base')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
            Baseline (Dist+Angle)
          </button>
          <button id="btn-model-gb" onclick="setActiveModel('xg_gradient_boosting')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
            HistGBM
          </button>
          <button id="btn-model-sb" onclick="setActiveModel('shot_statsbomb_xg')" class="px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition">
            StatsBomb
          </button>
        </div>
      </div>

      <!-- Filters Row -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Filter by Team:</label>
          <select id="team-filter" onchange="applyFilters()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500">
            <option value="ALL">All Teams (250 shots)</option>
          </select>
        </div>
        <div>
          <label class="block text-slate-400 mb-1">Filter by Outcome:</label>
          <select id="outcome-filter" onchange="applyFilters()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500">
            <option value="ALL">All Outcomes</option>
            <option value="Goal">Goals Only (20)</option>
            <option value="Saved">Saved Only</option>
            <option value="Blocked">Blocked Only</option>
            <option value="Off T">Off Target Only</option>
          </select>
        </div>
        <div>
          <div class="flex justify-between text-slate-400 mb-1">
            <span>Min Predicted xG:</span>
            <span id="xg-val-label" class="font-mono text-emerald-400 font-semibold">0.00</span>
          </div>
          <input type="range" id="xg-slider" min="0" max="0.8" step="0.02" value="0" oninput="updateXgSlider(this.value)" class="w-full accent-emerald-500 cursor-pointer">
        </div>
      </div>

      <!-- Pitch Container (StatsBomb Attacking Half X: 60 to 120, Y: 0 to 80) -->
      <div class="relative w-full aspect-[16/10] rounded-xl overflow-hidden border border-emerald-900/50">
        <svg id="pitch-svg" viewBox="58 0 64 80" class="w-full h-full pitch-svg" preserveAspectRatio="none">
          <!-- Pitch Markings -->
          <rect x="60" y="0" width="60" height="80" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- Halfway line -->
          <line x1="60" y1="0" x2="60" y2="80" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- Center Circle Arc -->
          <path d="M 60 30 A 10 10 0 0 1 60 50" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- Penalty Box (x: 102 to 120, y: 18 to 62) -->
          <rect x="102" y="18" width="18" height="44" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- 6-Yard Box (x: 114 to 120, y: 30 to 50) -->
          <rect x="114" y="30" width="6" height="20" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- Penalty Spot (x: 108, y: 40) -->
          <circle cx="108" cy="40" r="0.6" fill="rgba(255,255,255,0.6)"/>
          <!-- Penalty Arc -->
          <path d="M 102 33 A 10 10 0 0 0 102 47" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
          <!-- Goal Line & Goal (x: 120, y: 36 to 44) -->
          <line x1="120" y1="36" x2="120" y2="44" stroke="#ffffff" stroke-width="1.6"/>
          <rect x="120" y="36" width="2" height="8" fill="rgba(255,255,255,0.15)" stroke="rgba(255,255,255,0.5)" stroke-width="0.6"/>

          <!-- Dynamic Shots Layer -->
          <g id="shots-layer"></g>
        </svg>

        <!-- Floating Tooltip -->
        <div id="shot-tooltip" class="absolute pointer-events-none hidden bg-slate-950/95 border border-slate-700/80 p-3 rounded-xl shadow-2xl text-xs backdrop-blur-md z-30 min-w-[210px]">
          <div class="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-1.5">
            <span id="tt-player" class="font-bold text-white text-sm">Alexis Sánchez</span>
            <span id="tt-minute" class="text-slate-400 font-mono">45'</span>
          </div>
          <div class="space-y-1 text-slate-300">
            <div class="flex justify-between">
              <span class="text-slate-400">Team:</span>
              <span id="tt-team" class="font-medium text-white">Arsenal</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Outcome:</span>
              <span id="tt-outcome" class="font-bold text-emerald-400">Goal</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Distance / Angle:</span>
              <span id="tt-geometry" class="font-mono text-slate-200">14.2 yd / 28°</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Body Part:</span>
              <span id="tt-body" class="text-slate-200">Right Foot</span>
            </div>
            <div class="mt-2 pt-2 border-t border-slate-800">
              <div class="flex justify-between text-xs font-semibold">
                <span class="text-emerald-400">Predicted xG:</span>
                <span id="tt-xg" class="text-emerald-400 font-mono text-sm">0.342</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>StatsBomb Baseline:</span>
                <span id="tt-sb-xg" class="font-mono">0.318</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Pitch Legend -->
      <div class="flex flex-wrap items-center justify-between text-xs text-slate-400 mt-3 pt-3 border-t border-slate-800">
        <div class="flex items-center gap-4">
          <span class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]"></span> Goal (20)
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span> Saved
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-full bg-slate-500"></span> Blocked / Off Target
          </span>
        </div>
        <div class="text-[11px] text-slate-500">
          Circle size proportional to xG probability
        </div>
      </div>
    </div>

    <!-- Right Column: Model Calibration & Coefficients (4 cols) -->
    <div class="lg:col-span-4 flex flex-col gap-6">
      
      <!-- Calibration Breakdown Card -->
      <div class="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h3 class="font-bold text-white text-base flex items-center justify-between mb-3">
          <span>📈 Reliability & Calibration</span>
          <span class="text-xs font-normal text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-md border border-emerald-800/40">ECE 0.0269</span>
        </h3>
        <p class="text-xs text-slate-400 mb-4">Shots grouped by predicted probability vs. observed goal conversion rate.</p>

        <div id="calibration-bars" class="space-y-2.5">
          <!-- Dynamically populated -->
        </div>
      </div>

      <!-- Logistic Regression Coefficients Card -->
      <div class="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-lg flex-1">
        <h3 class="font-bold text-white text-base flex items-center justify-between mb-2">
          <span>🧠 What Drives xG?</span>
          <span class="text-xs font-normal text-slate-400 font-mono">Normalized Weights</span>
        </h3>
        <p class="text-xs text-slate-400 mb-3">Impact on log-odds of scoring (Full Logistic Regression).</p>

        <div id="coefficients-list" class="space-y-2 text-xs">
          <!-- Dynamically populated -->
        </div>
      </div>

    </div>
  </div>

  <!-- Bottom Table: Shot Event Explorer -->
  <div class="mt-6 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-lg">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
      <div>
        <h3 class="font-bold text-white text-base">📋 Held-Out Shot Event Explorer</h3>
        <p class="text-xs text-slate-400">Click any row to focus the shot marker on the pitch above.</p>
      </div>
      <div class="text-xs text-slate-400">
        Showing <span id="table-count" class="font-bold text-white">250</span> shots
      </div>
    </div>

    <div class="overflow-x-auto max-h-72 custom-scrollbar border border-slate-800/80 rounded-xl">
      <table class="w-full text-left text-xs text-slate-300">
        <thead class="bg-slate-950 text-slate-400 font-semibold sticky top-0 border-b border-slate-800">
          <tr>
            <th class="py-2.5 px-3">Minute</th>
            <th class="py-2.5 px-3">Player</th>
            <th class="py-2.5 px-3">Team</th>
            <th class="py-2.5 px-3">Outcome</th>
            <th class="py-2.5 px-3">Distance</th>
            <th class="py-2.5 px-3">Angle</th>
            <th class="py-2.5 px-3 text-emerald-400">Full LR xG</th>
            <th class="py-2.5 px-3 text-slate-400">Base xG</th>
            <th class="py-2.5 px-3 text-purple-400">HistGBM</th>
            <th class="py-2.5 px-3 text-cyan-400">StatsBomb</th>
          </tr>
        </thead>
        <tbody id="shots-table-body" class="divide-y divide-slate-800/40 font-mono">
          <!-- Dynamically populated -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- Data Payloads -->
  <script>
    const SHOTS = {json.dumps(shots_data)};
    const EVAL_DATA = {json.dumps(eval_data)};

    let activeModel = 'xg_lr_full';
    let filteredShots = [...SHOTS];

    // Populate Team Select
    const teamSelect = document.getElementById('team-filter');
    const teams = [...new Set(SHOTS.map(s => s.team))].sort();
    teams.forEach(t => {{
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      teamSelect.appendChild(opt);
    }});

    function setActiveModel(modelKey) {{
      activeModel = modelKey;
      ['lr-full', 'lr-base', 'gb', 'sb'].forEach(k => {{
        const btn = document.getElementById('btn-model-' + k);
        btn.className = 'px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition';
      }});
      if (modelKey === 'xg_lr_full') document.getElementById('btn-model-lr-full').className = 'px-3 py-1.5 rounded-lg bg-emerald-500 font-semibold text-white shadow-sm transition';
      if (modelKey === 'xg_lr_base') document.getElementById('btn-model-lr-base').className = 'px-3 py-1.5 rounded-lg bg-emerald-500 font-semibold text-white shadow-sm transition';
      if (modelKey === 'xg_gradient_boosting') document.getElementById('btn-model-gb').className = 'px-3 py-1.5 rounded-lg bg-purple-500 font-semibold text-white shadow-sm transition';
      if (modelKey === 'shot_statsbomb_xg') document.getElementById('btn-model-sb').className = 'px-3 py-1.5 rounded-lg bg-cyan-500 font-semibold text-white shadow-sm transition';

      renderPitchShots();
      renderTable();
    }}

    function updateXgSlider(val) {{
      document.getElementById('xg-val-label').textContent = parseFloat(val).toFixed(2);
      applyFilters();
    }}

    function applyFilters() {{
      const team = document.getElementById('team-filter').value;
      const outcome = document.getElementById('outcome-filter').value;
      const minXg = parseFloat(document.getElementById('xg-slider').value);

      filteredShots = SHOTS.filter(s => {{
        if (team !== 'ALL' && s.team !== team) return false;
        if (outcome !== 'ALL' && s.shot_outcome !== outcome) return false;
        const xg = s[activeModel] || 0;
        if (xg < minXg) return false;
        return true;
      }});

      renderPitchShots();
      renderTable();
    }}

    function renderPitchShots() {{
      const g = document.getElementById('shots-layer');
      g.innerHTML = '';
      const tooltip = document.getElementById('shot-tooltip');

      filteredShots.forEach(shot => {{
        const x = shot.location_x;
        const y = shot.location_y;
        if (!x || !y) return;

        const isGoal = shot.is_goal === 1;
        const xg = shot[activeModel] || 0.02;
        // Radius scale: min 0.8, max 3.2
        const r = Math.max(0.7, Math.min(3.2, Math.sqrt(xg) * 3.5));

        let fill = '#64748b';
        let stroke = '#94a3b8';
        if (isGoal) {{
          fill = '#10b981';
          stroke = '#34d399';
        }} else if (shot.shot_outcome === 'Saved') {{
          fill = '#06b6d4';
          stroke = '#22d3ee';
        }}

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', fill);
        circle.setAttribute('stroke', stroke);
        circle.setAttribute('stroke-width', '0.4');
        circle.setAttribute('opacity', isGoal ? '0.95' : '0.75');
        circle.setAttribute('class', 'shot-dot');
        circle.setAttribute('id', 'shot-node-' + shot.id);

        circle.addEventListener('mouseenter', (e) => {{
          showTooltip(e, shot);
        }});
        circle.addEventListener('mouseleave', () => {{
          tooltip.classList.add('hidden');
        }});

        g.appendChild(circle);
      }});
    }}

    function showTooltip(e, shot) {{
      const tooltip = document.getElementById('shot-tooltip');
      document.getElementById('tt-player').textContent = shot.player || 'Unknown Player';
      document.getElementById('tt-minute').textContent = shot.minute + "'";
      document.getElementById('tt-team').textContent = shot.team || 'Unknown';
      const outcomeEl = document.getElementById('tt-outcome');
      outcomeEl.textContent = shot.shot_outcome;
      outcomeEl.className = shot.is_goal === 1 ? 'font-bold text-emerald-400' : 'font-medium text-slate-300';
      document.getElementById('tt-geometry').textContent = `${{shot.distance_to_goal_yards}} yd / ${{shot.goal_angle_deg}}°`;
      document.getElementById('tt-body').textContent = shot.shot_body_part || 'Foot';
      document.getElementById('tt-xg').textContent = (shot[activeModel] || 0).toFixed(3);
      document.getElementById('tt-sb-xg').textContent = (shot.shot_statsbomb_xg || 0).toFixed(3);

      const svgBox = document.getElementById('pitch-svg').getBoundingClientRect();
      const parentBox = document.getElementById('pitch-svg').parentElement.getBoundingClientRect();
      
      const relX = e.clientX - parentBox.left;
      const relY = e.clientY - parentBox.top;

      tooltip.style.left = Math.min(parentBox.width - 230, Math.max(10, relX + 15)) + 'px';
      tooltip.style.top = Math.min(parentBox.height - 180, Math.max(10, relY - 40)) + 'px';
      tooltip.classList.remove('hidden');
    }}

    function highlightShot(shotId) {{
      const shot = SHOTS.find(s => s.id === shotId);
      if (!shot) return;
      const node = document.getElementById('shot-node-' + shotId);
      if (node) {{
        node.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        node.setAttribute('r', '4.5');
        node.setAttribute('stroke', '#ffffff');
        node.setAttribute('stroke-width', '1.0');
        setTimeout(() => {{
          renderPitchShots();
        }}, 2000);
      }}
    }}

    function renderTable() {{
      const tbody = document.getElementById('shots-table-body');
      tbody.innerHTML = '';
      document.getElementById('table-count').textContent = filteredShots.length;

      filteredShots.slice(0, 100).forEach(s => {{
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/60 cursor-pointer transition';
        tr.onclick = () => highlightShot(s.id);
        const goalBadge = s.is_goal === 1 ? '<span class="text-emerald-400 font-bold">Goal</span>' : s.shot_outcome;
        tr.innerHTML = `
          <td class="py-2 px-3 text-slate-400 font-sans">${{s.minute}}'</td>
          <td class="py-2 px-3 text-white font-medium font-sans">${{s.player}}</td>
          <td class="py-2 px-3 text-slate-300 font-sans">${{s.team}}</td>
          <td class="py-2 px-3 font-sans">${{goalBadge}}</td>
          <td class="py-2 px-3 text-slate-400">${{s.distance_to_goal_yards}} yd</td>
          <td class="py-2 px-3 text-slate-400">${{s.goal_angle_deg}}°</td>
          <td class="py-2 px-3 text-emerald-400 font-bold">${{(s.xg_lr_full || 0).toFixed(3)}}</td>
          <td class="py-2 px-3 text-slate-400">${{(s.xg_lr_base || 0).toFixed(3)}}</td>
          <td class="py-2 px-3 text-purple-400">${{(s.xg_gradient_boosting || 0).toFixed(3)}}</td>
          <td class="py-2 px-3 text-cyan-400">${{(s.shot_statsbomb_xg || 0).toFixed(3)}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // Render Calibration Bars
    function renderCalibration() {{
      const container = document.getElementById('calibration-bars');
      const lrFull = EVAL_DATA.find(m => m.model_name === "Logistic Regression (Full Features)");
      if (!lrFull) return;

      container.innerHTML = '';
      lrFull.calibration_bins.slice(0, 5).forEach(b => {{
        if (b.count === 0) return;
        const predPct = (b.avg_predicted_xg * 100).toFixed(1);
        const actualPct = (b.actual_goal_rate * 100).toFixed(1);
        
        const row = document.createElement('div');
        row.innerHTML = `
          <div class="flex justify-between text-[11px] mb-1">
            <span class="text-slate-400 font-mono">${{b.range}} (${{b.count}} shots)</span>
            <span>Pred: <strong class="text-emerald-400 font-mono">${{predPct}}%</strong> · Actual: <strong class="text-cyan-400 font-mono">${{actualPct}}%</strong></span>
          </div>
          <div class="w-full h-2 bg-slate-950 rounded-full overflow-hidden flex gap-0.5 p-0.5 border border-slate-800">
            <div class="h-full bg-emerald-500 rounded-full" style="width: ${{predPct}}%"></div>
          </div>
        `;
        container.appendChild(row);
      }});
    }}

    // Render Normalized Coefficients
    function renderCoefficients() {{
      const coefs = [
        {{ name: 'Distance to Goal', val: -0.8241, pos: false }},
        {{ name: 'Visible Goal Angle', val: +0.5081, pos: true }},
        {{ name: 'Header Attempt', val: -0.3592, pos: false }},
        {{ name: 'Volley Technique', val: -0.3569, pos: false }},
        {{ name: 'Cross Delivery', val: -0.3313, pos: false }},
        {{ name: 'First-Time Strike', val: +0.1666, pos: true }},
        {{ name: '1-on-1 with Keeper', val: +0.1314, pos: true }},
        {{ name: 'Through-Ball Pass', val: +0.0910, pos: true }}
      ];

      const container = document.getElementById('coefficients-list');
      container.innerHTML = '';
      coefs.forEach(c => {{
        const widthPct = Math.min(100, Math.abs(c.val) * 110);
        const color = c.pos ? 'bg-emerald-500' : 'bg-rose-500';
        const textColor = c.pos ? 'text-emerald-400' : 'text-rose-400';
        const sign = c.pos ? '+' : '';

        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 justify-between';
        item.innerHTML = `
          <span class="text-slate-300 w-32 truncate">${{c.name}}</span>
          <div class="flex-1 h-1.5 bg-slate-950 rounded-full overflow-hidden">
            <div class="${{color}} h-full rounded-full" style="width: ${{widthPct}}%"></div>
          </div>
          <span class="${{textColor}} font-mono w-14 text-right font-semibold">${{sign}}${{c.val.toFixed(2)}}</span>
        `;
        container.appendChild(item);
      }});
    }}

    // Init
    renderPitchShots();
    renderTable();
    renderCalibration();
    renderCoefficients();
  </script>
</body>
</html>
"""

target_file = ARTIFACT_DIR / "kick_dashboard.html"
with open(target_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Successfully generated standalone dashboard at: {target_file}")
