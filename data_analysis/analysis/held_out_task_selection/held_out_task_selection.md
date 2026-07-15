# Held-out task selection

Tasks ranked by novelty of tool usage/order/arguments relative to the remaining
train pool; every held-out tool stays covered by the train tasks.

| Task | Novelty | Tool JSD | Unseen 3-gram | Unseen args | Unseen vocab | Tools |
|---|---|---|---|---|---|---|
| garnish_cake | 0.116 | 0.137 | 0.000 | 0.000 | 0.498 | give_space, navigate_to_fixture, pick_up_object, place_on_object |
| meat_skewer_assembly | 0.114 | 0.110 | 0.000 | 0.000 | 0.537 | give_space, navigate_to_fixture, pick_up_object, place_in_receptacle |
| hot_dog_setup | 0.112 | 0.073 | 0.017 | 0.000 | 0.573 | give_space, navigate_to_fixture, open_hinged_part, pick_up_object, place_next_to, place_on_object |
| cluster_items_for_clearing | 0.092 | 0.110 | 0.002 | 0.000 | 0.391 | give_space, navigate_to_fixture, pick_up_object, place_next_to |
| beverage_organization | 0.076 | 0.134 | 0.002 | 0.000 | 0.237 | give_space, navigate_to_fixture, pick_up_object, place_on_surface |

## Higher-novelty candidates rejected

- `prepare_coffee`: tools not covered by >= 2 train tasks: ['place_under', 'press_button']
- `portion_yogurt`: tool set too similar to already selected garnish_cake
- `colorful_salsa`: tool set too similar to already selected garnish_cake
- `portion_hot_dogs`: tool set too similar to already selected garnish_cake
- `sweeten_coffee`: tool set too similar to already selected meat_skewer_assembly
- `align_silverware`: tool set too similar to already selected cluster_items_for_clearing
- `set_up_spice_station`: tool set too similar to already selected cluster_items_for_clearing
- `display_meat_variety`: tool set too similar to already selected meat_skewer_assembly
- `add_sugar_cubes`: tool set too similar to already selected garnish_cake
