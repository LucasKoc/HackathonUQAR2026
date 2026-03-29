# HackathonUQAR2026

## Description

This repository contains code for participation on IA'Hack 2026 (UQAR's annual Hackathon).
Domain of this Hackathon is about maritime.

## Installation

Venv creation and activation:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Dependency installation:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To launch the program with options:

```bash
python main.py
# Or
python main.py <Train_Data_Path> <Test_Data_Path>
```

## Data source

- Watkins Marine Mammal Sound Database
  - 70% Train / 30% Test
  - 100 samples .wav
