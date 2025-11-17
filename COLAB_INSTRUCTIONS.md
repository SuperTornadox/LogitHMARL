# Running VARIANT-1 Experiment on Google Colab A100

## Quick Start

1. **Upload the entire project folder to Google Drive**
   - Upload the `LogitHMARL` folder to your Google Drive

2. **Open Google Colab and create a new notebook**
   - Go to https://colab.research.google.com/
   - Create a new notebook
   - Set runtime to A100 GPU: Runtime → Change runtime type → A100 GPU

3. **Mount Google Drive and navigate to project**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.chdir('/content/drive/MyDrive/LogitHMARL')  # Adjust path as needed
   !pwd
   ```

4. **Install dependencies**
   ```python
   !pip install -q torch torchvision
   !pip install -q stable-baselines3
   !pip install -q gymnasium
   !pip install -q pandas matplotlib seaborn
   !pip install -q imageio
   ```

5. **Run the experiment**
   ```python
   # Set environment variable for full mode
   import os
   os.environ['MODE'] = 'full'

   # Run the experiment
   !python run_experiments.py
   ```

6. **Or run in background with nohup**
   ```bash
   %%bash
   export MODE=full
   nohup python run_experiments.py > variant1_colab.log 2>&1 &
   echo "Experiment started in background"
   ```

7. **Monitor progress**
   ```python
   # Check log file
   !tail -50 variant1_colab.log

   # Check GPU usage
   !nvidia-smi

   # Check if process is running
   !ps aux | grep python
   ```

## Expected Runtime on A100

- **Single method (NL-HMARL)**: ~30-60 minutes
- **All 9 methods**: ~4-6 hours total
- Much faster than CPU-only local run (10-20+ hours)

## VARIANT-1 Configuration

The following features are automatically enabled when MODE=full:

1. **Deadline Pressure (Exponential Decay)**
   - 30% urgent orders with tight deadlines
   - Exponential value degradation (e^(-3*(t-D)/D))

2. **Zone Capacity Constraints**
   - Zone 0,1,3: max 3 agents
   - Zone 2: max 4 agents
   - Exponential congestion penalties

3. **Heavy Task Emphasis**
   - 40% heavy items, 35% medium, 25% light
   - 6 forklifts (30% of 20 pickers)
   - Lower weight thresholds (forklift_only: 70kg vs 90kg)

4. **Burst Arrivals**
   - 25% probability of bursts
   - 8-20 orders per burst
   - 75% zone correlation

5. **Large-Scale Training**
   - 200,000 training steps
   - 16 parallel environments
   - Batch size: 2048
   - 300 evaluation steps

6. **Reproducibility**
   - Fixed random seed: 42

## Checking Results

```python
# View results summary
import pandas as pd
results = pd.read_csv('results/results.csv')
print(results.sort_values('total_value', ascending=False))

# Download results to local machine
from google.colab import files
files.download('results/results.csv')

# Download all visualizations
!zip -r results.zip results/
files.download('results.zip')
```

## Alternative: One-Line Colab Cell

```python
# Complete setup and run in one cell
!git clone YOUR_REPO_URL LogitHMARL  # If using git
# OR use Drive mount as shown above

%cd LogitHMARL
!pip install -q torch stable-baselines3 gymnasium pandas matplotlib seaborn imageio

import os
os.environ['MODE'] = 'full'
!python run_experiments.py
```

## Monitoring Tips

1. **Real-time monitoring**: Keep the notebook open and run `!tail -f variant1_colab.log` in a cell
2. **Prevent disconnection**: Keep the browser tab active or use Colab Pro
3. **Save checkpoints**: The system auto-saves results to `results/` folder
4. **Backup**: Periodically copy results to Drive to prevent data loss

## After Completion

The experiment will generate:
- `results/results.csv` - Performance metrics for all 9 methods
- `results/order_arrival_pattern.png` - Visualization of burst patterns
- `results/[method_name]/` - Individual method results and visualizations
- Training logs showing convergence
