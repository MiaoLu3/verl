# ALFWorld Trajectory Viewer

Local Flask app for browsing ALFWorld trajectory dumps under
`/scratch/m000069-pm05/miaolu/verl/trajectories/`.

## Run

```bash
# Conda env that already has flask installed:
conda activate /scratch/m000069-pm05/miaolu/conda_env/verl-agent
# (or: pip install flask)

cd /scratch/m000069-pm05/miaolu/verl/tools/traj_viewer
python app.py --port 5000
```

The app binds to `127.0.0.1:5000` (localhost only).

## Access from your laptop

SSH tunnel:

```bash
ssh -L 5000:localhost:5000 <cluster>
# then in a browser: http://localhost:5000
```

## Pages

- `/` — list of all run directories
- `/run/<run_name>/` — paginated, filterable, sortable trajectory table
  (50/page; filter by Won/Lost/All; sort by any column)
- `/run/<run_name>/traj/<filename>` — full trajectory detail with every
  turn; `<think>` blocks are shown in muted italic, `<action>` blocks are
  highlighted. Warning banner appears if the response was truncated
  mid-`<think>`. `full_decoded_sequence` is only loaded if you click the
  "Show full_decoded_sequence" button.

## Caching

First open of a run indexes all `*.jsonl` files in that directory (only
top-level fields are read; the heavy `turns` / `full_decoded_sequence`
lists are skipped). For ~8k files expect ~30-60s. Subsequent opens and
pagination are instant. The cache is per-process in memory; restart the
app to rebuild.

## Deps

- `flask` (only external dep)

## Stack

Python + Flask + Jinja2 + vanilla HTML/CSS. No build step, no JS
framework.
