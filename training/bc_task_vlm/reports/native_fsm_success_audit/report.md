# Native `_check_success()` versus symbolic FSM audit

## Technical summary

The audit originally covered 52 canonical verified TaskSpecs. During follow-up,
`PrepareSandwichStation` was found to have been accidentally omitted from the
flat verified inventory and was promoted after two independent 12/12
all-configuration canaries. The audit now covers all **53 canonical verified
TaskSpecs** currently present in
`data_generation/task_level/tasks/specs/verified`. Every native RoboCasa
`_check_success()` implementation was compared with the corresponding symbolic
initial state, task preconditions, effects, and terminal goal conditions.

At the time of the initial audit, **49/52 tasks were represented consistently under the
current symbolic action abstraction**. The audit found three exceptions; the
two non-pruning errors have now been corrected:

- **OrganizeCondiments — pruning-related gap.** Native success requires the
  distractor to remain on the counter and the gripper to be far from it. The
  distractor is absent from symbolic state, absent from the FSM goal, and a
  live-evaluation special case deliberately removes those native predicates.
- **GarnishCake — resolved non-pruning error.** Native success permits the
  one selected cherry to be on either `cake` or `cake_plate`. The FSM requires
  exactly one cherry on `cake` and zero cherries on `cake_plate`, rejecting a
  native-valid outcome explicitly allowed by the task instruction. The FSM now
  uses an exact count across both allowed locations.
- **PrepareCoffee — resolved non-pruning error.** Native success requires the
  mug under the dispenser, the machine on, and physical gripper clearance. The
  FSM terminal goal checks only `coffee_machine.turned_on`; its `press_button`
  effect sets that flag without requiring the mug to be under the dispenser.
  Thus a symbolic trajectory can press the button without ever placing the mug
  and still satisfy the FSM. The button effect and terminal goal now both
  require the mug under the dispenser.

The hypothesis that every mismatch came from pruning was therefore false. After
the two fixes, the remaining documented semantic exception is the intentionally
pruned OrganizeCondiments distractor.

## Scope and definitions

The unit of audit is one canonical verified task specification. “Aligned” means
that the FSM represents the same semantic outcome as native success after
accounting for the existing symbolic tools. It does **not** mean numerical or
physical equivalence with MuJoCo.

The audit distinguishes:

- **Semantic mismatch:** the FSM accepts a native-invalid semantic outcome or
  rejects a native-valid semantic outcome.
- **Pruning-related mismatch:** native success references an entity deliberately
  absent from the trajectory-compatible environment or symbolic state.
- **Physical abstraction difference:** the FSM represents a geometric or
  temporal predicate with a semantic tool/effect, such as `place_next_to`,
  “stirred,” or an object-location relation.

## Cross-cutting physical differences

Native checkers commonly require `gripper_obj_far`, exact XY distances,
multi-frame contact, or simulator-specific containment. The FSM does not model
continuous robot/object geometry. Relevant objects cannot remain symbolically
held when their goal locations are satisfied, and placement tools are intended
to realize the corresponding physical relation. These differences explain why
native and FSM success are recorded separately, but they are not pruning.

Two cases deserve continued simulator-side monitoring:

- **ArrangeBreadBowl:** native success additionally checks contact between the
  two breads. The FSM requires both breads inside the bowl but not bread-to-bread
  contact.
- **CheeseMixing:** native success detects physical stirring for multiple
  simulation frames and verifies the pot is on a specific burner. The symbolic
  FSM represents this through the fixed initial pot placement and a semantic
  stirring effect.

These are intentional abstraction differences unless live replay shows that
the semantic tools do not reliably realize the native predicates.

## Task-by-task result

| Task | Classification | Evidence summary |
|---|---|---|
| AddLemonToFish | Aligned | Lemon, fish, plate, and counter relations represented; native names are mapped to symbolic names. |
| AddSugarCubes | Aligned | Both cubes are on the cake plate; cake and plate preservation are preconditions. |
| AlcoholServingPrep | Aligned | Alcohol and cup end on the dining surface. |
| AlignSilverware | Aligned via spatial abstraction | Left/right alignment is represented by semantic machine flags. |
| ArrangeBreadBowl | Aligned with physical caveat | Both breads and bowl locations match; bread-to-bread contact remains physical-only. |
| ArrangeDrinkware | Aligned | Pitcher and cup destination relations match. |
| ArrangeTea | Aligned | Kettle/mug tray placement and final closed cabinet match. |
| BeverageOrganization | Aligned | All sampled drinks end on the dining surface; cardinality synchronization handles pruning. |
| BowlAndCup | Aligned | Cup-in-bowl and bowl-on-kitchen-counter relations match. |
| CandleCleanup | Aligned | Both objects in cabinet and final cabinet closure match. |
| CheeseMixing | Aligned via dynamic abstraction | Cheese/pot relation and stirring are represented symbolically; physical stirring remains native-only. |
| ClusterItemsForClearing | Aligned via spatial abstraction | Counter contact and clustering are represented by object locations and flags. |
| ColorfulSalsa | Aligned | Four ingredients on the fixed cutting board match. |
| CondimentCollection | Aligned | Both condiments inside the cabinet match. |
| CutBuffetPizza | Aligned | Cutter destination and pizza preservation match. |
| DateNight | Aligned | Alcohol and decoration destination relations match. |
| DeliverStraw | Aligned | Straw-in-glass relation matches. |
| DessertAssembly | Aligned | Bowl/cupcake-on-tray and preservation of tray/dessert nesting match. |
| DisplayMeatVariety | Aligned | Three meats in fixed serving tray match; dining-table naming is normalized. |
| DistributeChicken | Aligned | Exactly one drumstick per plate is represented by counts. |
| GarnishCake | **Aligned after fix** | Exact count is evaluated across both native-valid cherry targets. |
| GarnishCupcake | Aligned via spatial abstraction | Cinnamon proximity and chocolate/plate relations match. |
| GarnishPancake | Aligned | Strawberry/pancake/plate nesting and counter preservation match. |
| GatherMarinadeIngredients | Aligned via spatial abstraction | Garlic containment and two proximity relations match. |
| HotDogSetup | Aligned via spatial abstraction | Bun/sausage containment and condiment proximity match. |
| LemonSeasoningFish | Aligned | Lemon-on-cutting-board relation matches. |
| LineUpCondiments | Aligned via spatial abstraction | Counter placement, stove proximity, and lineup relation match. |
| MatchCupAndDrink | Aligned via spatial abstraction | Each bottle-to-cup proximity relation matches. |
| MeatSkewerAssembly | Aligned | Both skewers in the fixed oven tray match. |
| OrganizeCondiments | **Pruning-related mismatch** | Native distractor predicate is deliberately omitted from symbolic state and live native checking. |
| PlateStoreDinner | Aligned | Exclusive steak split plus nested bowl-in-fridge outcome matches. |
| PortionHotDogs | Aligned | Exactly one bun and sausage per fixed plate match. |
| PortionYogurt | Aligned | Exactly two yogurts per fixed plate match. |
| PrepareCheeseStation | Aligned via spatial abstraction | Cheese and grater proximity to the bowl match. |
| PrepareCocktailStation | Aligned via spatial abstraction | Lemon containment and liquor/glass proximity match. |
| PrepareCoffee | **Aligned after fix** | Mug-under-dispenser is required before activation and at terminal success. |
| PrepareDrinkStation | Aligned via spatial abstraction | Cup/mug containment and pitcher proximity match. |
| PrepareSausageCheese | Aligned | Sausage and cheese on the fixed cutting board match. |
| PrepareSandwichStation | Aligned via spatial abstraction | Bowl contents are preserved and both bowl and baguette are staged near the toaster oven. |
| PrepareSoupServing | Aligned | Ladle-in-pot and final cabinet closure match. |
| SeasoningSteak | Aligned via spatial abstraction | Shaker surface and steak proximity relations match. |
| ServeMealJuice | Aligned via spatial abstraction | One cup assigned near each distinct plate matches. |
| ServeSteak | Aligned | Pan-on-table followed by steak-on-plate outcome matches. |
| SetBowlsForSoup | Aligned | Each bowl is assigned to a different plate. |
| SetupBowls | Aligned via spatial abstraction | Distinct stool assignments and counter contact match. |
| SetupButterPlate | Aligned via spatial abstraction | Butter containment and knife proximity match. |
| SetupSodaBowl | Aligned | Both sodas in the fixed ice bowl match. |
| SetUpSpiceStation | Aligned via spatial abstraction | Three objects on the adjacent counter near the stove match. |
| SetupWineGlasses | Aligned via spatial abstraction | Each glass is assigned near a different plate. |
| SpicyMarinade | Aligned | Bowl/condiment counter placement and lime/garlic board placement match. |
| SweetenCoffee | Aligned via spatial abstraction | Sugar-in-coffee and milk-near-coffee relations match. |
| TongBuffetSetup | Aligned via spatial abstraction | Tongs on dining counter near the fixed food tray match. |
| VeggieDipPrep | Aligned | Cucumber, carrot, and bowl in the fixed tray match. |

## Recommended next steps

1. Keep the current pruning policy unchanged for now, as requested.
2. Treat OrganizeCondiments as a documented pruning exception rather than
   claiming exact native/FSM equivalence for that task.
3. Add an automated source-to-spec audit inventory to CI so every verified task
   has exactly one readable native checker and an auditable symbolic payload.
4. Continue reporting FSM and native success separately for physical abstraction
   differences; do not reinterpret them as model-learning failures.

## Limitations and robustness

This is a static semantic audit. It reads all 53 native checker implementations
and all 53 symbolic specifications, but it does not prove that teleportation and
semantic manipulation tools satisfy every continuous MuJoCo predicate across
all layouts. Existing or future live replay remains the appropriate check for
that physical layer. The machine-readable source inventory is saved in
`evidence.json` beside this report.
