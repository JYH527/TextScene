---
name: parameter-filling-reasoning
description: Fill structured physical scenario parameters step by step using explicit rules. Use when populating environment, time, entity, and action fields from a physical scenario or when parameter values are unstable or need validation.
---

# Parameter Filling Reasoning

## Purpose
Turn a structured physical scenario into a consistent parameter set for OpenSCENARIO 1.0 generation.

## Core principles
- Fill parameters in a fixed order.
- Prefer explicit rules over free-form reasoning.
- Do not guess values when the input is missing; use the fallback rule defined for that field.
- Keep entity fields internally consistent, especially road/lane/speed/action dependencies.
- Validate the result before returning it.

## Standard workflow

### Step 1. Fill `environment`
Fill all environment fields first:
1. `environment.cloud_state`
2. `environment.precipitation_type`
3. `environment.precipitation_intensity`
4. `environment.fog_visual_range`
5. `environment.sun_azimuth`
6. `environment.sun_elevation`
7. `environment.sun_intensity`

Rules:
- Use the field definitions in `references.md`.
- If a field depends on weather type, fill the weather type first.
- If multiple values are plausible, choose the one that best matches the source scene description and the reference rule.

### Step 2. Fill `simulation_duration`
Set the simulation duration after environment is determined.

Rules:
- Use the field definitions in `references.md`
- Respect the duration unit used by the project.

### Step 3. Fill `entities`
Fill all entities using the same rule set.

Recommended order:
1. `Entity.name`
2. `Entity.is_ego`
3. `Entity.vehicle_model`
4. `Entity.init_road_id`
5. `Entity.init_lane_id`
6. `Entity.init_s`
7. `Entity.init_speed`
8. Other entity fields required by the reference rules

Rules:
- Use the field definitions in `references.md`
- Fill `Ego` first, then other entities.
- If one field constrains another, fill the constraint source first.

### Step 4. Fill `entity.actions`
Fill actions after entity initialization is complete.

Recommended order:
1. `Entity.actions.type`
2. `Entity.actions.params`
3. `Entity.actions.trigger_type`
4. `Entity.actions.trigger_value`
5. `Entity.actions.trigger_ref`

Rules:
- Use the field definitions in `references.md`
- Determine the action type before filling its parameters.
- Fill trigger fields only after the action meaning is clear.
- Ensure trigger references point to valid entities, lanes, distances, or events defined in the scene.

## Validation checklist
Before finalizing, verify:
- [ ] Every required field has a value
- [ ] No field violates the reference rules
- [ ] Entity fields are consistent with each other
- [ ] Trigger references are valid and resolvable
- [ ] No field was filled with unsupported assumptions

If validation fails, revise the earliest conflicting field and re-fill downstream fields.

## Output expectations
- Return the filled parameters in the same structure as the input.
- Keep naming consistent with the project schema.
- If a value cannot be determined, mark it explicitly according to the project rule rather than inventing one.

## References
See `references.md` for the filling rules of each parameter and `skills.md` for the workflow order.