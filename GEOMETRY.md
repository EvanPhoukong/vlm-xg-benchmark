# Checking image geometry

`geometry_diagnostic.py` evaluates how well GPT-6 Luna reads the existing images.
It does not train the VLM, change the images, or produce new xG predictions.

From the repository folder, with the virtual environment active and
`OPENAI_API_KEY` set:

```bash
python geometry_diagnostic.py --input results/gpt_6_luna/shots.csv --n 50 --seed 7 --output results/geometry_gpt_6_luna_50
```

This makes one sequential request per selected image. Completed responses are
saved. Repeating the same command reuses them. A different prompt, model,
image, or source dataset requires a new output folder.

The 50 shots are sampled uniformly from the saved 300-shot run after sorting by
shot ID. Selection does not use goals, xG, or previous model errors.
The model receives the unchanged image, marker meanings, the coordinate scale,
and the definition of goal-opening angle. It receives no true shot coordinates,
goalkeeper coordinates, outcomes, or xG values.

The test measures:

- Distance to goal center and angle between the two goalposts, as directly
  estimated by the VLM.
- Shooter and goalkeeper location errors.
- Distance and angle calculated in code from the VLM's reported shooter location.
  This separates visual localization from the numerical calculation.

Pitch units use the benchmark's normalized StatsBomb 120×80 coordinates; they
are not stadium measurements in meters or verified physical yards. In the
vertical image, lateral position is `StatsBomb y - 40`, and depth from the goal
line is `120 - StatsBomb x`. Goal-opening angle is reported in degrees.

Tolerances are fixed in advance: 2 pitch units for distance and position, and
5 degrees for angle. The report includes all eligible shots in tolerance rates;
abstentions count as failures. Coordinate errors among returned answers are
reported alongside coverage. Goalkeepers absent from the reference data or
outside the image area are excluded from goalkeeper errors and counted.
Overlapping markers can make a known keeper location hard to see.

The output folder includes a report, two figures, per-shot predictions, metrics,
the selected IDs, reference freeze frames, and request/image hashes. The saved
cache contains model answers and usage information, never API credentials.

To prepare only the reference data without OpenAI requests:

```bash
python geometry_diagnostic.py --prepare-only --n 50 --seed 7 --output results/geometry_gpt_6_luna_50
```

Run the offline checks:

```bash
python -m unittest discover -s tests -v
```

An explicit geometry prompt can perform differently from the earlier xG prompt.
This diagnostic can suggest the next experiment, but cannot establish the cause
of the original xG errors or demonstrate an improvement in xG performance.
