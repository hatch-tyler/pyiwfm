# Calibration Test Fixtures

Small SMP and weights files for testing IWFM2OBS and CalcTypHyd implementations.
These fixtures can be run through both the Fortran executables and pyiwfm to verify
identical output.

## Files

| File | Purpose |
|------|---------|
| `obs_gw.smp` | 5 bores, 10 obs each (dummy observed values) |
| `sim_gw.smp` | 5 bores, 24 monthly simulated values each |
| `expected_gw_out.smp` | Expected IWFM2OBS output (np.interp linear interpolation) |
| `cluster_weights.txt` | 5 wells, 2 clusters |
| `water_levels.smp` | 5 wells, 24 monthly records (2 years) |
| `expected_typhyd_cls0.txt` | Expected CalcTypHyd output for cluster 0 |
| `expected_typhyd_cls1.txt` | Expected CalcTypHyd output for cluster 1 |

## Bore IDs

C2VSimFG-style naming with `%` character to verify special character handling:
- `S_380313N1219426W001%1`
- `S_375204N1214521W001%1`
- `S_381045N1220102W001%1`
- `S_374830N1213900W001%1`
- `S_380600N1218000W001%1`

## Sim Value Patterns

- Bore 1: linear ramp 100 + 1*i (i=0..23)
- Bore 2: decreasing 200 - 0.5*i
- Bore 3: constant 50.0
- Bore 4: steep ramp 150 + 2*i
- Bore 5: alternating 80 +/- 5

## Running Through Fortran Executables

1. Copy `obs_gw.smp` and `sim_gw.smp` to a working directory
2. Run IWFM2OBS:
   ```
   IWFM2OBS_x64.exe obs_gw.smp sim_gw.smp GW_OUT.smp
   ```
3. Compare `GW_OUT.smp` against `expected_gw_out.smp` (should match to 4 decimal places)

For CalcTypHyd:
1. Copy `water_levels.smp` and `cluster_weights.txt` to a working directory
2. Create a CalcTypHyd input file referencing these
3. Run: `CalcTypeHyd_x64.exe CalcTypeHyd.in`
4. Compare output against `expected_typhyd_cls*.txt`
