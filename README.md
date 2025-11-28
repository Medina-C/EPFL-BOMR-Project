# BOMR Final Project: Group 08

## Overview

This document describes the recommended procedure for setting up the software environment required to run the BOMR Final Project. To ensure full reproducibility and avoid dependency conflicts, the project should be executed in an isolated Python environment.

## Environment Setup

Python 3.13 is required. Both Anaconda and the standard Python venv module are supported.

### Option A: Conda (recommended)
```
conda create -n <env_name> python=3.13
conda activate <env_name>
```

### Option B: Python venv
```
python -m venv <env_name>
```

### Activate the virtual environment

**macOS/Linux:**
```
source <env_name>/bin/activate
```

**Windows:**
```
<env_name>\Scripts\activate
```

Replace `<env_name>` with your preferred environment name in all commands above.

## Installing Dependencies

Once the virtual environment is active, install all project requirements:

```
pip install -r requirements.txt
```

All experiments and results were obtained using the pinned dependency versions specified in `requirements.txt.`

## Running the Demo 

TODO