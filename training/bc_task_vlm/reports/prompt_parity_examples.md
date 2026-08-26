# Exact initial-turn SFT versus live-evaluation prompts

These are generated from real rendered trajectories. The tool schemas are supplied separately to the model, so they are included in full.

## Demonstration-generation request

This is the exact current production-style first-attempt prompt for `PrepareCoffee`: structured-random sampling, temperature 0.6, random starts and access state, tick format, `simplified_v3`, and no forced partition. Gemini also receives the response schema printed after the text prompt.

### Generator text prompt

```text
Structured random configuration: communication_message_length=low, communication_message_complexity=high, tool_call_diversity=high

Use the fixed configuration above for this single trajectory.
- Return exactly one trajectory using the normal single-trajectory JSON format requested below.
- Do not return a samples array, a responses array, or multiple trajectory candidates.
- communication_message_length controls whether communicate messages are terse, sentence-length, or detailed.
- communication_message_complexity controls whether communicate messages are simple status/request messages, include task constraints, or include multi-part rationale and contingency planning.
- tool_call_diversity controls whether the trajectory uses a narrow, moderate, or broad task-relevant mix of allowed non-communication tools.
- Keep all tool calls task-relevant; do not add unnecessary actions just to increase diversity.

You are simulating 2 cooperative robot agents in a physical kitchen.
Generate one valid trajectory for PrepareCoffee as a list of simultaneous ticks.

Eight rules:
1. Every tick must contain both agent fields. Give an unblocked agent exactly one real tool call. After an agent calls wait_for_signal once, write {"state":"blocked"} for it on each later tick while the wait is unresolved; never repeat the wait call. Keep the blocked marker on the release tick, then give the awakened agent a real call on the following tick. Set the top-level format to "explicit_blocked_v1".
2. The sampled coordinator is agent_0. Tick 0: agent_0 proposes one concrete division of work with coordination_phase="propose" while the partner uses "await_plan". Tick 1: agent_0 uses "await_confirmation" while the partner uses "confirm". Start physical work on tick 2.
3. Choose an efficient division of work from the initial state. Give both agents useful work, keep each object's pickup and placement with one owner, avoid unnecessary handovers, and state the chosen exact IDs in the opening proposal.
4. Track each agent's current fixture. navigate_to_fixture(X) sets its location to X; give_space(X) removes it from X. Every pickup, placement, fixture-part, and control action requires the matching current fixture. After give_space, navigate again before acting there. An agent may hold at most one object and may place only that object.
5. A tick is atomic. The agents may work together at a roomy counter on different objects, but may not use the same object or the same exclusive cabinet, drawer, appliance, sink, or stove workspace in one tick. If one agent calls give_space(X), the other may enter or use X only on the following tick, never during that give_space tick.
6. Use wait_for_signal only when an agent truly cannot do useful independent work. The holder is whichever agent currently uses the resource; the waiter is the other agent that needs it next, and these roles may switch later. To change ownership of an exclusive fixture: the waiter requests it, then waits on the next tick. The holder finishes, calls give_space, and later sends a matching release. A release takes effect only after its tick finishes: never wait and release in the same tick, keep the waiter blocked throughout the release tick, and let it resume only on the following tick. An ordinary message does not wake it.
7. Communicate only a plan, request, release, correction, or one portion_complete message; never claim global completion. If an agent finishes its assigned physical calls while its partner still has calls remaining, it stays active: on its next call send exactly one communicate with coordination_phase="portion_complete" saying only its own portion is done; on the immediately following tick call wait_for_signal(from=<partner>, about=<one exact object or fixture in the partner's remaining work>). It then remains blocked until a later matching release makes it active on the following tick. If the partner reaches the FSM goal in the wait tick, end the episode without a release. If the first required message after physical work is a resource release, combine it with portion_complete in that one communicate call by including both coordination_phase="portion_complete" and releases="<resource id>".
8. Use only the tools and exact symbolic IDs below. Required args must appear. Optional args appear only under the condition stated by that tool; otherwise omit them. Every real tool call needs one short first-person reasoning sentence. Output JSON only.

Exclusive-fixture handover example:
- tick N: agent_0 communicates a request for fridge; agent_1 finishes useful fridge work.
- tick N+1: agent_0 calls wait_for_signal(from=agent_1, about=fridge); agent_1 calls give_space(fridge).
- tick N+2: agent_0 is {"state":"blocked"}; agent_1 communicates to agent_0 with releases=fridge.
- tick N+3: agent_0 calls navigate_to_fixture(fridge); agent_1 emits another real call unless blocked.
After give_space(X), that agent is no longer at X and must navigate before acting there again.

Task-specific facts and sampled work assignment:
- Open cab.hinged before using pick_up_object from cab.
- The coffee machine only turns on if the mug is placed under the dispenser before pressing the button.

Composite task:
- PrepareCoffee
- Goal: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.

Initial agent positions:
- agent_0: coffee_machine
- agent_1: cab

Initial task state:
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "coffee_machine"
    },
    "agent_1": {
      "held_object": null,
      "location": "cab"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    },
    "coffee_machine": {
      "controls": {
        "start_button": {
          "control_type": "button",
          "state": "off"
        }
      },
      "fixture_type": "coffee_machine",
      "parts": {}
    }
  },
  "machine_state": {
    "coffee_machine": {
      "turned_on": false
    },
    "prepare_coffee": {}
  },
  "objects": {
    "mug": {
      "location": "cab",
      "object_type": "mug"
    }
  }
}

Tool contracts:
communicate
  Purpose: Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. args.coordination_phase is required for protocol messages and must be one of propose, await_plan, await_confirmation, confirm, or portion_complete. Omit it from ordinary requests, corrections, handoffs, and informational messages.
  Required: to, message
  Optional (otherwise omit):
    releases: one exact object or fixture ID; only when this message hands that resource to an agent already waiting for it
    coordination_phase: only for propose, await_plan, await_confirmation, confirm, or portion_complete protocol messages

wait_for_signal
  Purpose: Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
  Required: from, about

navigate_to_fixture
  Purpose: Move the robot base to the working pose of a fixture.
  Required: fixture_id
  allowed_fixture_ids: cab, coffee_machine

pick_up_object
  Purpose: Grasp and lift an object from a symbolic source region.
  Required: object_id, source_id
  allowed_object_ids: mug
  allowed_source_ids: cab

place_under
  Purpose: Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below.
  Required: object_id, reference_fixture_id
  allowed_object_ids: mug
  allowed_reference_fixture_ids: coffee_machine

press_button
  Purpose: Activate a discrete button-like control on a fixture.
  Required: target_id, control_id
  allowed_control_ids: start_button
  allowed_target_ids: coffee_machine

give_space
  Purpose: Yield working room at a fixture so the other agent can complete an action there.
  Required: fixture_id
  allowed_fixture_ids: cab, coffee_machine

open_hinged_part
  Purpose: Open a hinged door, lid, or articulated head on a fixture.
  Required: target_id, part_id
  allowed_part_ids: hinged
  allowed_target_ids: cab

Variation key: traj-000000-attempt-00
```
### Generator response schema

```json
{
  "properties": {
    "format": {
      "enum": [
        "explicit_blocked_v1"
      ],
      "type": "string"
    },
    "ticks": {
      "items": {
        "properties": {
          "agent_0": {
            "anyOf": [
              {
                "properties": {
                  "args": {
                    "properties": {
                      "coordination_phase": {
                        "enum": [
                          "propose",
                          "await_plan",
                          "await_confirmation",
                          "confirm",
                          "portion_complete"
                        ],
                        "type": "STRING"
                      },
                      "message": {
                        "type": "STRING"
                      },
                      "releases": {
                        "type": "STRING"
                      },
                      "to": {
                        "enum": [
                          "agent_1"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "to",
                      "message"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "communicate"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "about": {
                        "type": "STRING"
                      },
                      "from": {
                        "enum": [
                          "agent_1"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "from",
                      "about"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "wait_for_signal"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "fixture_id": {
                        "enum": [
                          "cab",
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "navigate_to_fixture"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "object_id": {
                        "enum": [
                          "mug"
                        ],
                        "type": "STRING"
                      },
                      "source_id": {
                        "enum": [
                          "cab"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "object_id",
                      "source_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "pick_up_object"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "object_id": {
                        "enum": [
                          "mug"
                        ],
                        "type": "STRING"
                      },
                      "reference_fixture_id": {
                        "enum": [
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "object_id",
                      "reference_fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "place_under"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "control_id": {
                        "enum": [
                          "start_button"
                        ],
                        "type": "STRING"
                      },
                      "target_id": {
                        "enum": [
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "target_id",
                      "control_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "press_button"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "fixture_id": {
                        "enum": [
                          "cab",
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "give_space"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "part_id": {
                        "enum": [
                          "hinged"
                        ],
                        "type": "STRING"
                      },
                      "target_id": {
                        "enum": [
                          "cab"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "target_id",
                      "part_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "open_hinged_part"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "state": {
                    "enum": [
                      "blocked"
                    ],
                    "type": "string"
                  }
                },
                "required": [
                  "state"
                ],
                "type": "object"
              }
            ],
            "description": "What agent_0 does at this tick. This field is required whenever agent_0 is not blocked by an earlier wait_for_signal; omit it only while that wait remains blocked."
          },
          "agent_1": {
            "anyOf": [
              {
                "properties": {
                  "args": {
                    "properties": {
                      "coordination_phase": {
                        "enum": [
                          "propose",
                          "await_plan",
                          "await_confirmation",
                          "confirm",
                          "portion_complete"
                        ],
                        "type": "STRING"
                      },
                      "message": {
                        "type": "STRING"
                      },
                      "releases": {
                        "type": "STRING"
                      },
                      "to": {
                        "enum": [
                          "agent_0"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "to",
                      "message"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "communicate"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "about": {
                        "type": "STRING"
                      },
                      "from": {
                        "enum": [
                          "agent_0"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "from",
                      "about"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "wait_for_signal"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "fixture_id": {
                        "enum": [
                          "cab",
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "navigate_to_fixture"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "object_id": {
                        "enum": [
                          "mug"
                        ],
                        "type": "STRING"
                      },
                      "source_id": {
                        "enum": [
                          "cab"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "object_id",
                      "source_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "pick_up_object"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "object_id": {
                        "enum": [
                          "mug"
                        ],
                        "type": "STRING"
                      },
                      "reference_fixture_id": {
                        "enum": [
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "object_id",
                      "reference_fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "place_under"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "control_id": {
                        "enum": [
                          "start_button"
                        ],
                        "type": "STRING"
                      },
                      "target_id": {
                        "enum": [
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "target_id",
                      "control_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "press_button"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "fixture_id": {
                        "enum": [
                          "cab",
                          "coffee_machine"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "fixture_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "give_space"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "args": {
                    "properties": {
                      "part_id": {
                        "enum": [
                          "hinged"
                        ],
                        "type": "STRING"
                      },
                      "target_id": {
                        "enum": [
                          "cab"
                        ],
                        "type": "STRING"
                      }
                    },
                    "required": [
                      "target_id",
                      "part_id"
                    ],
                    "type": "OBJECT"
                  },
                  "reasoning": {
                    "type": "STRING"
                  },
                  "tool": {
                    "enum": [
                      "open_hinged_part"
                    ],
                    "type": "STRING"
                  }
                },
                "required": [
                  "tool",
                  "args",
                  "reasoning"
                ],
                "type": "object"
              },
              {
                "properties": {
                  "state": {
                    "enum": [
                      "blocked"
                    ],
                    "type": "string"
                  }
                },
                "required": [
                  "state"
                ],
                "type": "object"
              }
            ],
            "description": "What agent_1 does at this tick. This field is required whenever agent_1 is not blocked by an earlier wait_for_signal; omit it only while that wait remains blocked."
          },
          "tick": {
            "description": "0-based instant; both agents act on it.",
            "type": "integer"
          }
        },
        "required": [
          "tick",
          "agent_0",
          "agent_1"
        ],
        "type": "object"
      },
      "type": "array"
    }
  },
  "required": [
    "ticks",
    "format"
  ],
  "type": "object"
}
```

## The Qwen chat template adds the missing output-format instructions

The earlier view separated the user message and function schemas, but the model sees both after Qwen's chat template inserts this wrapper into the system turn. Therefore the tool-call format is present in the actual tokenized request even though it is absent from the user message.

### Model-visible tool wrapper

```text
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{one JSON function schema per available tool}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

## ArrangeBreadBowl

Carrier: `arrange_bread_bowl/traj_000002`

### System message (both paths)

```text
You are a robot task planner. Predict exactly one next tool call for the current acting agent. Request the camera views you need before an action by calling get_image. Use images only as scene context. Do not output image paths or describe the images.
```

### SFT user message

```text
Task family: ArrangeBreadBowl
Task instruction: Pick up the heated bread from the toaster oven and place it in the bowl. Then move the bread bowl to the dining counter.
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Previous executed symbolic action history:
- none

Available tools for this task:
- communicate(to, message): Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.
- wait_for_signal(from, about): Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
- navigate_to_fixture(fixture_id): Move the robot base to the working pose of a fixture. Constraints: allowed_fixture_ids=[toaster_oven, counter, dining_counter].
- open_hinged_part(target_id, part_id): Open a hinged door, lid, or articulated head on a fixture. Constraints: allowed_part_ids=[door]; allowed_target_ids=[toaster_oven].
- pick_up_object(object_id, source_id): Grasp and lift an object from a symbolic source region. Constraints: allowed_object_ids=[toaster_oven_bread, bowl]; allowed_source_ids=[toaster_oven, counter].
- place_in_receptacle(object_id, one of (target_id | receptacle_id)): Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters. Constraints: allowed_object_ids=[toaster_oven_bread]; allowed_receptacle_ids=[bowl].
- place_on_surface(object_id, one of (target_id | support_id)): Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id. Constraints: allowed_object_ids=[bowl]; allowed_target_ids=[dining_counter].
- give_space(fixture_id): Yield working room at a fixture so the other agent can complete an action there. Constraints: allowed_fixture_ids=[toaster_oven, counter, dining_counter].
- get_image(views): Request one or more camera views before choosing the next action. Constraints: allowed_views=[top_view, room_view, map, wrist, agentview_center, agentview_left, agentview_right].


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done. On your next tick, call wait_for_signal about a real object or fixture in the partner's outstanding work. Stay blocked until the partner sends a later matching release.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the allowed tools listed above.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Live-evaluation user message

```text
Task family: ArrangeBreadBowl
Task instruction: ArrangeBreadBowl
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Previous executed symbolic action history:
- none

Available tools for this task:
- communicate(to, message): Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.
- wait_for_signal(from, about): Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
- navigate_to_fixture(fixture_id): Move the robot base to the working pose of a fixture. Constraints: allowed_fixture_ids=[toaster_oven, counter, dining_counter].
- open_hinged_part(target_id, part_id): Open a hinged door, lid, or articulated head on a fixture. Constraints: allowed_part_ids=[door]; allowed_target_ids=[toaster_oven].
- pick_up_object(object_id, source_id): Grasp and lift an object from a symbolic source region. Constraints: allowed_object_ids=[toaster_oven_bread, bowl]; allowed_source_ids=[toaster_oven, counter].
- place_in_receptacle(object_id, one of (target_id | receptacle_id)): Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters. Constraints: allowed_object_ids=[toaster_oven_bread]; allowed_receptacle_ids=[bowl].
- place_on_surface(object_id, one of (target_id | support_id)): Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id. Constraints: allowed_object_ids=[bowl]; allowed_target_ids=[dining_counter].
- give_space(fixture_id): Yield working room at a fixture so the other agent can complete an action there. Constraints: allowed_fixture_ids=[toaster_oven, counter, dining_counter].
- get_image(views): Request one or more camera views before choosing the next action. Constraints: allowed_views=[top_view, room_view, map, wrist, agentview_center, agentview_left, agentview_right].


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done. On your next tick, call wait_for_signal about a real object or fixture in the partner's outstanding work. Stay blocked until the partner sends a later matching release.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the allowed tools listed above.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Function/tool schemas (both paths)

```json
[
  {
    "function": {
      "description": "Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.",
      "name": "communicate",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "coordination_phase": {
            "enum": [
              "propose",
              "await_plan",
              "await_confirmation",
              "confirm",
              "portion_complete"
            ],
            "type": "string"
          },
          "message": {
            "type": "string"
          },
          "releases": {
            "type": "string"
          },
          "to": {
            "enum": [
              "agent_0",
              "agent_1"
            ],
            "type": "string"
          }
        },
        "required": [
          "to",
          "message"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.",
      "name": "wait_for_signal",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "about": {
            "type": "string"
          },
          "from": {
            "type": "string"
          }
        },
        "required": [
          "from",
          "about"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Move the robot base to the working pose of a fixture.",
      "name": "navigate_to_fixture",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "enum": [
              "toaster_oven",
              "counter",
              "dining_counter"
            ],
            "type": "string"
          }
        },
        "required": [
          "fixture_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Open a hinged door, lid, or articulated head on a fixture.",
      "name": "open_hinged_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "enum": [
              "door"
            ],
            "type": "string"
          },
          "target_id": {
            "enum": [
              "toaster_oven"
            ],
            "type": "string"
          }
        },
        "required": [
          "target_id",
          "part_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Grasp and lift an object from a symbolic source region.",
      "name": "pick_up_object",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "enum": [
              "toaster_oven_bread",
              "bowl"
            ],
            "type": "string"
          },
          "source_id": {
            "enum": [
              "toaster_oven",
              "counter"
            ],
            "type": "string"
          },
          "source_site_id": {
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "source_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters. Requires at least one of: target_id or receptacle_id.",
      "name": "place_in_receptacle",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "enum": [
              "toaster_oven_bread"
            ],
            "type": "string"
          },
          "receptacle_id": {
            "enum": [
              "bowl"
            ],
            "type": "string"
          },
          "relative_position": {
            "type": "string"
          },
          "target_id": {
            "type": "string"
          },
          "target_site_id": {
            "type": "string"
          }
        },
        "required": [
          "object_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id. Requires at least one of: target_id or support_id.",
      "name": "place_on_surface",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "enum": [
              "bowl"
            ],
            "type": "string"
          },
          "relative_position": {
            "type": "string"
          },
          "support_id": {
            "type": "string"
          },
          "target_id": {
            "enum": [
              "dining_counter"
            ],
            "type": "string"
          },
          "target_site_id": {
            "type": "string"
          }
        },
        "required": [
          "object_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Yield working room at a fixture so the other agent can complete an action there.",
      "name": "give_space",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "enum": [
              "toaster_oven",
              "counter",
              "dining_counter"
            ],
            "type": "string"
          }
        },
        "required": [
          "fixture_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Request one or more camera views before choosing the next action.",
      "name": "get_image",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "views": {
            "items": {
              "enum": [
                "top_view",
                "room_view",
                "map",
                "wrist",
                "agentview_center",
                "agentview_left",
                "agentview_right"
              ],
              "type": "string"
            },
            "type": "array"
          }
        },
        "required": [
          "views"
        ],
        "type": "object"
      }
    },
    "type": "function"
  }
]
```

### Saved initial state (not included in either model prompt)

```json
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "counter"
    },
    "agent_1": {
      "held_object": null,
      "location": "dining_counter"
    }
  },
  "fixtures": {
    "counter": {
      "fixture_type": "counter"
    },
    "dining_counter": {
      "fixture_type": "dining_counter"
    },
    "toaster_oven": {
      "fixture_type": "toaster_oven",
      "parent_fixture": "counter",
      "parts": {
        "door": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    }
  },
  "machine_state": {
    "arrange_bread_bowl": {}
  },
  "objects": {
    "bowl": {
      "location": "counter",
      "object_type": "bowl"
    },
    "bowl_bread": {
      "location": "bowl",
      "object_type": "bread_food"
    },
    "toaster_oven_bread": {
      "location": "toaster_oven",
      "object_type": "bread_food"
    }
  }
}
```

## PrepareCoffee

Carrier: `prepare_coffee/traj_000000`

### System message (both paths)

```text
You are a robot task planner. Predict exactly one next tool call for the current acting agent. Request the camera views you need before an action by calling get_image. Use images only as scene context. Do not output image paths or describe the images.
```

### SFT user message

```text
Task family: PrepareCoffee
Task instruction: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Previous executed symbolic action history:
- none

Available tools for this task:
- communicate(to, message): Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.
- wait_for_signal(from, about): Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
- navigate_to_fixture(fixture_id): Move the robot base to the working pose of a fixture. Constraints: allowed_fixture_ids=[cab, coffee_machine].
- pick_up_object(object_id, source_id): Grasp and lift an object from a symbolic source region. Constraints: allowed_object_ids=[mug]; allowed_source_ids=[cab].
- place_under(object_id, one of (target_id | reference_fixture_id)): Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below. Constraints: allowed_object_ids=[mug]; allowed_reference_fixture_ids=[coffee_machine].
- press_button(target_id, control_id): Activate a discrete button-like control on a fixture. Constraints: allowed_control_ids=[start_button]; allowed_target_ids=[coffee_machine].
- give_space(fixture_id): Yield working room at a fixture so the other agent can complete an action there. Constraints: allowed_fixture_ids=[cab, coffee_machine].
- open_hinged_part(target_id, part_id): Open a hinged door, lid, or articulated head on a fixture. Constraints: allowed_part_ids=[hinged]; allowed_target_ids=[cab].
- get_image(views): Request one or more camera views before choosing the next action. Constraints: allowed_views=[top_view, room_view, map, wrist, agentview_center, agentview_left, agentview_right].


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done. On your next tick, call wait_for_signal about a real object or fixture in the partner's outstanding work. Stay blocked until the partner sends a later matching release.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the allowed tools listed above.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Live-evaluation user message

```text
Task family: PrepareCoffee
Task instruction: PrepareCoffee
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Previous executed symbolic action history:
- none

Available tools for this task:
- communicate(to, message): Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.
- wait_for_signal(from, about): Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
- navigate_to_fixture(fixture_id): Move the robot base to the working pose of a fixture. Constraints: allowed_fixture_ids=[cab, coffee_machine].
- pick_up_object(object_id, source_id): Grasp and lift an object from a symbolic source region. Constraints: allowed_object_ids=[mug]; allowed_source_ids=[cab].
- place_under(object_id, one of (target_id | reference_fixture_id)): Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below. Constraints: allowed_object_ids=[mug]; allowed_reference_fixture_ids=[coffee_machine].
- press_button(target_id, control_id): Activate a discrete button-like control on a fixture. Constraints: allowed_control_ids=[start_button]; allowed_target_ids=[coffee_machine].
- give_space(fixture_id): Yield working room at a fixture so the other agent can complete an action there. Constraints: allowed_fixture_ids=[cab, coffee_machine].
- open_hinged_part(target_id, part_id): Open a hinged door, lid, or articulated head on a fixture. Constraints: allowed_part_ids=[hinged]; allowed_target_ids=[cab].
- get_image(views): Request one or more camera views before choosing the next action. Constraints: allowed_views=[top_view, room_view, map, wrist, agentview_center, agentview_left, agentview_right].


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done. On your next tick, call wait_for_signal about a real object or fixture in the partner's outstanding work. Stay blocked until the partner sends a later matching release.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the allowed tools listed above.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Function/tool schemas (both paths)

```json
[
  {
    "function": {
      "description": "Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. coordination_phase is optional and must be omitted from ordinary messages. Use it only for the opening propose/await_plan/await_confirmation/confirm handshake or for portion_complete.",
      "name": "communicate",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "coordination_phase": {
            "enum": [
              "propose",
              "await_plan",
              "await_confirmation",
              "confirm",
              "portion_complete"
            ],
            "type": "string"
          },
          "message": {
            "type": "string"
          },
          "releases": {
            "type": "string"
          },
          "to": {
            "enum": [
              "agent_0",
              "agent_1"
            ],
            "type": "string"
          }
        },
        "required": [
          "to",
          "message"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.",
      "name": "wait_for_signal",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "about": {
            "type": "string"
          },
          "from": {
            "type": "string"
          }
        },
        "required": [
          "from",
          "about"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Move the robot base to the working pose of a fixture.",
      "name": "navigate_to_fixture",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "enum": [
              "cab",
              "coffee_machine"
            ],
            "type": "string"
          }
        },
        "required": [
          "fixture_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Grasp and lift an object from a symbolic source region.",
      "name": "pick_up_object",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "enum": [
              "mug"
            ],
            "type": "string"
          },
          "source_id": {
            "enum": [
              "cab"
            ],
            "type": "string"
          },
          "source_site_id": {
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "source_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below. Requires at least one of: target_id or reference_fixture_id.",
      "name": "place_under",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "enum": [
              "mug"
            ],
            "type": "string"
          },
          "reference_fixture_id": {
            "enum": [
              "coffee_machine"
            ],
            "type": "string"
          },
          "target_id": {
            "type": "string"
          },
          "target_site_id": {
            "type": "string"
          }
        },
        "required": [
          "object_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Activate a discrete button-like control on a fixture.",
      "name": "press_button",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "control_id": {
            "enum": [
              "start_button"
            ],
            "type": "string"
          },
          "target_id": {
            "enum": [
              "coffee_machine"
            ],
            "type": "string"
          }
        },
        "required": [
          "target_id",
          "control_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Yield working room at a fixture so the other agent can complete an action there.",
      "name": "give_space",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "enum": [
              "cab",
              "coffee_machine"
            ],
            "type": "string"
          }
        },
        "required": [
          "fixture_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Open a hinged door, lid, or articulated head on a fixture.",
      "name": "open_hinged_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "enum": [
              "hinged"
            ],
            "type": "string"
          },
          "target_id": {
            "enum": [
              "cab"
            ],
            "type": "string"
          }
        },
        "required": [
          "target_id",
          "part_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Request one or more camera views before choosing the next action.",
      "name": "get_image",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "views": {
            "items": {
              "enum": [
                "top_view",
                "room_view",
                "map",
                "wrist",
                "agentview_center",
                "agentview_left",
                "agentview_right"
              ],
              "type": "string"
            },
            "type": "array"
          }
        },
        "required": [
          "views"
        ],
        "type": "object"
      }
    },
    "type": "function"
  }
]
```

### Saved initial state (not included in either model prompt)

```json
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "coffee_machine"
    },
    "agent_1": {
      "held_object": null,
      "location": "cab"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "open"
        }
      }
    },
    "coffee_machine": {
      "controls": {
        "start_button": {
          "control_type": "button",
          "state": "off"
        }
      },
      "fixture_type": "coffee_machine",
      "parts": {}
    }
  },
  "machine_state": {
    "coffee_machine": {
      "turned_on": false
    },
    "prepare_coffee": {}
  },
  "objects": {
    "mug": {
      "location": "cab",
      "object_type": "mug"
    }
  }
}
```
