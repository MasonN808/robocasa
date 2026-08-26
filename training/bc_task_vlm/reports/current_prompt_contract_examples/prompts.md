# Exact demonstration-generation, SFT, and live-evaluation prompts

These examples are rendered through the current production prompt builders and real rendered trajectory carriers. Tool schemas are supplied separately to the model and are therefore printed alongside the user prompts.

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
1. Every tick must contain both agent fields. Give an unblocked agent exactly one real tool call. BLOCKED is not an action and cannot start a wait. Use {"state":"blocked"} only after that same agent has called wait_for_signal and before its matching release arrives. Write the blocked marker on each later tick while the wait is unresolved; never repeat the wait call. Keep the blocked marker on the release tick, then give the awakened agent a real call on the following tick. Set the top-level format to "explicit_blocked_v1".
2. The sampled coordinator is agent_0. Tick 0: agent_0 proposes one concrete division of work with coordination_phase="propose" while the partner uses "await_plan". Tick 1: agent_0 uses "await_confirmation" while the partner uses "confirm". Start physical work on tick 2.
3. Choose an efficient division of work from the initial state. Prefer giving both agents a coherent physical responsibility when the task has multiple objects or separable actions. Keep each object's pickup and placement with one owner. A single agent may do all physical work only when the task is one inherently sequential chain or splitting it would create work that does not help complete the goal. State the chosen exact IDs in the opening proposal.
4. Track each agent's current fixture. navigate_to_fixture(X) sets its location to X; give_space(X) removes it from X. Every pickup, placement, fixture-part, and control action requires the matching current fixture. Being near or able to see fixture X does not count as being at X. Before using an object, container, or appliance at X, call navigate_to_fixture(X) unless the agent's current symbolic location is already X. After give_space, navigate again before acting there. An agent can hold only one object at a time. There is no tool for passing an object between agents: the agent that picks up an object must place that same object itself before picking up another. A cabinet and its parent counter are one shared workspace: navigate to the parent counter and use cabinet tools from there. An agent at an exclusive child appliance may use its parent counter and objects on that counter without navigating again; it remains at the child. Opening or closing a fixture part requires empty hands: if holding an object, place it at a valid temporary location first. When practical, open the required fixture before picking up the object.
5. A tick is atomic. The agents may work together at a roomy counter or cabinet workspace on different objects, but may not use the same object or the same exclusive drawer, appliance, sink, or stove workspace in one tick. A cabinet door may not open or close during another agent's access to that cabinet's contents. give_space(X) frees X only after its tick finishes: the partner must not enter or use X during the give_space tick. An unblocked partner may enter or use X on the following tick without waiting for a release.
6. wait_for_signal has only two uses. To hand over an exclusive fixture X, the incoming agent communicates `When done, release "X"` and on its very next call waits with about=X. The current user finishes at X, calls give_space(X), and on a later tick communicates with releases=X. Use the fixture ID X, not the ID of an object at X. Waiting does not move an agent, and releasing an object at X does not free X. A finished agent follows the same visible-request rule: communicate `My portion is done. Release "X" if you need me again` with coordination_phase="portion_complete", then on its very next call wait with about=X. The other agent cannot see the wait call. In every case, the communicate call immediately before wait must name the exact same X and say to release it; never omit the preceding message or put another action between it and wait. A waiter stays blocked through the matching release tick and resumes on the following tick. Never wait and release in the same tick; an ordinary message does not wake it.
7. Communicate only a plan, request, release, correction, or one portion_complete message; never claim global completion. If an agent finishes its assigned physical calls while its partner still has calls remaining, it stays active: on its next call send exactly one communicate with coordination_phase="portion_complete" saying only its own portion is done, using the finished-portion message from rule 6; on the immediately following tick make the matching wait call. If the opening plan assigns an agent no physical actions, its first call after the handshake is that finished-portion message and its following call is the matching wait. If the partner's action on the same tick as portion_complete satisfies the FSM goal, end the episode on that tick: do not add a wait or any later tick. It then remains blocked until a later matching release makes it active on the following tick. If the partner reaches the FSM goal in the wait tick, end the episode without a release. If the first required message after physical work is a resource release, combine it with portion_complete in that one communicate call by including both coordination_phase="portion_complete" and releases="<resource id>".
8. Use only the tools and exact symbolic IDs below. Required args must appear. Optional args appear only under the condition stated by that tool; otherwise omit them. Every real tool call needs one short first-person reasoning sentence. Output JSON only.

Exclusive-fixture handover example:
- tick N: agent_0 communicates `When done, release "fridge"`; agent_1 finishes useful fridge work.
- tick N+1: agent_0 calls wait_for_signal(from=agent_1, about=fridge); agent_1 calls give_space(fridge).
- tick N+2: agent_0 is {"state":"blocked"}; agent_1 communicates to agent_0 with releases=fridge and coordination_phase=portion_complete if this was agent_1's final physical work.
- tick N+3: agent_0 calls navigate_to_fixture(fridge); agent_1 calls wait_for_signal about one real object or fixture in agent_0's remaining work if agent_1 is finished.
After give_space(X), that agent is no longer at X and must navigate before acting there again.

Task-specific facts and work-division guidance:
- Open cab.hinged before using pick_up_object from cab.
- Only use press_button after mug is already at coffee_machine.
- The coffee machine only turns on if the mug is placed under the dispenser before pressing the button.

Composite task:
- PrepareCoffee
- Goal: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.

Initial agent positions:
- agent_0: cabinet_parent_counter
- agent_1: coffee_machine

Initial task state:
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "cabinet_parent_counter"
    },
    "agent_1": {
      "held_object": null,
      "location": "coffee_machine"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parent_fixture": "cabinet_parent_counter",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    },
    "cabinet_parent_counter": {
      "controls": {},
      "fixture_type": "counter",
      "parts": {}
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

Concurrency facts (derived from the same fixture-type policy as the concurrent validator):
- Exclusive, one agent at a time: coffee_machine.
- Roomy/shared for simultaneous work on different objects: cab, cabinet_parent_counter.

Placement-anchor vocabulary (use the exact field shown; never use target_id as an alias):
- place_under: reference_fixture_id = under a fixture or appliance.

Tool contracts:
communicate
  Purpose: Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. args.coordination_phase is required for protocol messages and must be one of propose, await_plan, await_confirmation, confirm, or portion_complete. Omit it from ordinary requests, corrections, handoffs, and informational messages.
  Required:
    to (STRING): Exact ID of the agent receiving the message.
    message (STRING): Short natural-language coordination message for the other agent.
  Optional (otherwise omit):
    releases (STRING): Optional exact symbolic object or fixture ID being handed over. Include only for a real handover; omit from all other messages.
    coordination_phase (STRING): Optional opening-protocol phase, or portion_complete when reporting only this agent's assigned work complete. Omit for ordinary messages.

wait_for_signal
  Purpose: Pause until the other agent hands over what you are waiting for. args.about names that thing by its exact symbolic id -- an object id or a fixture id that appears in initial_state, copied exactly. It is a THING you are waiting FOR, never an event you are waiting ON: wait about='coffee_machine', never about='mug_placed' or 'your_turn'. An id that is not in initial_state can never be released, so the wait blocks forever and the whole plan is thrown away. You stay paused until args.from sends a communicate whose args.releases names the same id -- their other messages do not wake you, so you do not need to wait again after an unrelated one.
  Required:
    from (STRING): Exact ID of the agent whose matching release will end the wait.
    about (STRING): Exact symbolic object or fixture ID being awaited.

navigate_to_fixture
  Purpose: Move the robot base to the working pose of a fixture.
  Required:
    fixture_id (STRING): Exact symbolic ID of the fixture or workspace.

pick_up_object
  Purpose: Grasp and lift an object from a symbolic source region.
  Required:
    object_id (STRING): Exact symbolic ID of the movable object being manipulated.
    source_id (STRING): Exact symbolic ID of the object's current source region.

place_under
  Purpose: Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below.
  Required:
    object_id (STRING): Exact symbolic ID of the movable object being manipulated.
    reference_fixture_id (STRING): Exact symbolic ID of the fixture used as the placement reference.

press_button
  Purpose: Activate a discrete button-like control on a fixture.
  Required:
    target_id (STRING): Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.
    control_id (STRING): Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.

give_space
  Purpose: Leave a workspace when vacating it is genuinely needed so another agent can enter or work there. This is a handover action, not an idle, completion, or participation action. Roomy counters, islands, and dining tables normally need no give_space when agents use different objects.
  Required:
    fixture_id (STRING): Exact symbolic ID of the fixture or workspace.

open_hinged_part
  Purpose: Open a hinged door, lid, or articulated head on a fixture.
  Required:
    target_id (STRING): Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.
    part_id (STRING): Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).

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
                          "cabinet_parent_counter",
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
                          "cabinet_parent_counter",
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
                          "cabinet_parent_counter",
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
                          "cabinet_parent_counter",
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

Carrier: `arrange_bread_bowl/traj_000000`

### System message (both paths)

```text
You are a robot task planner. Predict exactly one next tool call for the current acting agent. Request the camera views you need before an action by calling get_image. Use images only as scene context. Do not output image paths or describe the images.
```

### SFT user message

```text
Task family: ArrangeBreadBowl
Task instruction: pick up the heated bread from the toaster oven, place it in the bowl on the counter, then move the bowl containing both breads to the dining counter.
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Symbolic initial state:
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

Concurrency facts (derived from the same fixture-type policy as the concurrent validator):
- Exclusive, one agent at a time: toaster_oven.
- Roomy/shared for simultaneous work on different objects: counter, dining_counter.

Previous executed symbolic action history:
- none


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done and naming one exact release keyword X, for example `My portion is done. Release "X" if you need me again`. On your very next call, call wait_for_signal with about=X. The private wait call is not visible to the partner, so the message must name the same X and say to release it. Stay blocked until the partner sends a later matching release.

Physical workspace handoffs:
- give_space(X) frees X only after that tool call finishes. The partner must not enter or use X in the same concurrent tick.
- An unblocked partner may enter or use X on the following tick without calling wait_for_signal and without receiving a release.
- Use wait_for_signal only when the agent must actually block before X becomes free. Once waiting, it remains blocked throughout a later matching release tick and resumes only on the following tick. Never wait and receive the matching release in the same tick.
Workspace relationships:
- A cabinet and its parent counter are one shared workspace. Navigate to the parent counter for cabinet work. Cabinet tools may be used from that counter, and different agents may use different objects there without a handover. Do not open or close the cabinet during another agent's access to its contents.
- If an agent is at an exclusive child appliance on a parent counter, it may use that counter and objects on it without navigating again. This does not move the agent: its location remains the exclusive child appliance.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- Symbolic IDs are dictionary keys in the initial state; copy the relevant object, fixture, part, or control key verbatim. Use a site ID only when the state explicitly provides a named site.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the provided function schemas.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Live-evaluation user message

```text
Task family: ArrangeBreadBowl
Task instruction: pick up the heated bread from the toaster oven, place it in the bowl on the counter, then move the bowl containing both breads to the dining counter.
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Symbolic initial state:
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

Concurrency facts (derived from the same fixture-type policy as the concurrent validator):
- Exclusive, one agent at a time: toaster_oven.
- Roomy/shared for simultaneous work on different objects: counter, dining_counter.

Previous executed symbolic action history:
- none


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done and naming one exact release keyword X, for example `My portion is done. Release "X" if you need me again`. On your very next call, call wait_for_signal with about=X. The private wait call is not visible to the partner, so the message must name the same X and say to release it. Stay blocked until the partner sends a later matching release.

Physical workspace handoffs:
- give_space(X) frees X only after that tool call finishes. The partner must not enter or use X in the same concurrent tick.
- An unblocked partner may enter or use X on the following tick without calling wait_for_signal and without receiving a release.
- Use wait_for_signal only when the agent must actually block before X becomes free. Once waiting, it remains blocked throughout a later matching release tick and resumes only on the following tick. Never wait and receive the matching release in the same tick.
Workspace relationships:
- A cabinet and its parent counter are one shared workspace. Navigate to the parent counter for cabinet work. Cabinet tools may be used from that counter, and different agents may use different objects there without a handover. Do not open or close the cabinet during another agent's access to its contents.
- If an agent is at an exclusive child appliance on a parent counter, it may use that counter and objects on it without navigating again. This does not move the agent: its location remains the exclusive child appliance.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- Symbolic IDs are dictionary keys in the initial state; copy the relevant object, fixture, part, or control key verbatim. Use a site ID only when the state explicitly provides a named site.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the provided function schemas.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Function/tool schemas (both paths)

```json
[
  {
    "function": {
      "description": "Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. args.coordination_phase is required for protocol messages and must be one of propose, await_plan, await_confirmation, confirm, or portion_complete. Omit it from ordinary requests, corrections, handoffs, and informational messages.",
      "name": "communicate",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "coordination_phase": {
            "description": "Optional opening-protocol phase, or portion_complete when reporting only this agent's assigned work complete. Omit for ordinary messages.",
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
            "description": "Short natural-language coordination message for the other agent.",
            "type": "string"
          },
          "releases": {
            "description": "Optional exact symbolic object or fixture ID being handed over. Include only for a real handover; omit from all other messages.",
            "type": "string"
          },
          "to": {
            "description": "Exact ID of the agent receiving the message.",
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
            "description": "Exact symbolic object or fixture ID being awaited.",
            "type": "string"
          },
          "from": {
            "description": "Exact ID of the agent whose matching release will end the wait.",
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
      "description": "Close a hinged door, lid, or articulated head on a fixture.",
      "name": "close_hinged_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Push in a sliding drawer or rack on a fixture.",
      "name": "close_sliding_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Capture one or more images from named views for later inspection.",
      "name": "get_image",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "views": {
            "description": "One or more camera-view names to request.",
            "items": {
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
  },
  {
    "function": {
      "description": "Leave a workspace when vacating it is genuinely needed so another agent can enter or work there. This is a handover action, not an idle, completion, or participation action. Roomy counters, islands, and dining tables normally need no give_space when agents use different objects.",
      "name": "give_space",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "description": "Exact symbolic ID of the fixture or workspace.",
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
      "description": "Move the robot base to the working pose of a fixture.",
      "name": "navigate_to_fixture",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "description": "Exact symbolic ID of the fixture or workspace.",
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
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Pull out a sliding drawer or rack on a fixture.",
      "name": "open_sliding_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "source_id": {
            "description": "Exact symbolic ID of the object's current source region.",
            "type": "string"
          },
          "source_site_id": {
            "description": "Optional exact sub-site within the source; omit when no named sub-site matters.",
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
      "description": "Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters.",
      "name": "place_in_receptacle",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "receptacle_id": {
            "description": "Exact symbolic ID of the destination receptacle.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "receptacle_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object adjacent to a reference object or nearby fixture on the same support surface. Use reference_object_id only for movable objects and reference_fixture_id only for fixtures. When using reference_fixture_id, the agent may navigate either to that reference fixture or to its declared adjacent_location_id before placing. Requires at least one of: reference_object_id or reference_fixture_id.",
      "name": "place_next_to",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "reference_fixture_id": {
            "description": "Exact symbolic ID of the fixture used as the placement reference.",
            "type": "string"
          },
          "reference_object_id": {
            "description": "Exact symbolic ID of the movable object used as the placement reference.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
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
      "description": "Place an object on top of another movable support object such as a plate or cutting_board.",
      "name": "place_on_object",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "support_object_id": {
            "description": "Exact symbolic ID of the movable object supporting the placed object.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "support_object_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id.",
      "name": "place_on_surface",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "support_id": {
            "description": "Exact symbolic ID of the fixture or surface supporting the placement.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "support_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below.",
      "name": "place_under",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "reference_fixture_id": {
            "description": "Exact symbolic ID of the fixture used as the placement reference.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "reference_fixture_id"
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
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Activate a discrete lever-like control on a fixture.",
      "name": "press_lever",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "control_id": {
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Set a rotary or handle-like control to an explicit goal state.",
      "name": "set_rotary_control",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "control_id": {
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "goal": {
            "description": "Requested final symbolic state of the control.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
            "type": "string"
          }
        },
        "required": [
          "target_id",
          "control_id",
          "goal"
        ],
        "type": "object"
      }
    },
    "type": "function"
  }
]
```

### Saved initial state (included identically in both model prompts)

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

Symbolic initial state:
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "cabinet_parent_counter"
    },
    "agent_1": {
      "held_object": null,
      "location": "coffee_machine"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parent_fixture": "cabinet_parent_counter",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    },
    "cabinet_parent_counter": {
      "controls": {},
      "fixture_type": "counter",
      "parts": {}
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

Concurrency facts (derived from the same fixture-type policy as the concurrent validator):
- Exclusive, one agent at a time: coffee_machine.
- Roomy/shared for simultaneous work on different objects: cab, cabinet_parent_counter.

Previous executed symbolic action history:
- none


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done and naming one exact release keyword X, for example `My portion is done. Release "X" if you need me again`. On your very next call, call wait_for_signal with about=X. The private wait call is not visible to the partner, so the message must name the same X and say to release it. Stay blocked until the partner sends a later matching release.

Physical workspace handoffs:
- give_space(X) frees X only after that tool call finishes. The partner must not enter or use X in the same concurrent tick.
- An unblocked partner may enter or use X on the following tick without calling wait_for_signal and without receiving a release.
- Use wait_for_signal only when the agent must actually block before X becomes free. Once waiting, it remains blocked throughout a later matching release tick and resumes only on the following tick. Never wait and receive the matching release in the same tick.
Workspace relationships:
- A cabinet and its parent counter are one shared workspace. Navigate to the parent counter for cabinet work. Cabinet tools may be used from that counter, and different agents may use different objects there without a handover. Do not open or close the cabinet during another agent's access to its contents.
- If an agent is at an exclusive child appliance on a parent counter, it may use that counter and objects on it without navigating again. This does not move the agent: its location remains the exclusive child appliance.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- Symbolic IDs are dictionary keys in the initial state; copy the relevant object, fixture, part, or control key verbatim. Use a site ID only when the state explicitly provides a named site.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the provided function schemas.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Live-evaluation user message

```text
Task family: PrepareCoffee
Task instruction: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.
Current acting agent: agent_0
Coordinator for this episode: agent_0
Observation views attached in order: none

Symbolic initial state:
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "cabinet_parent_counter"
    },
    "agent_1": {
      "held_object": null,
      "location": "coffee_machine"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parent_fixture": "cabinet_parent_counter",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    },
    "cabinet_parent_counter": {
      "controls": {},
      "fixture_type": "counter",
      "parts": {}
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

Concurrency facts (derived from the same fixture-type policy as the concurrent validator):
- Exclusive, one agent at a time: coffee_machine.
- Roomy/shared for simultaneous work on different objects: cab, cabinet_parent_counter.

Previous executed symbolic action history:
- none


Coordinator handshake:
- Before physical work, the coordinator proposes one concrete division using exact symbolic IDs and coordination_phase=propose.
- On its simultaneous opening turn, the other agent uses coordination_phase=await_plan rather than proposing independently.
- After receiving the proposal, the other agent confirms it with coordination_phase=confirm while the coordinator uses coordination_phase=await_confirmation.
- Begin physical work only after the confirmation has been delivered.
- Never claim that the global task is complete; the FSM alone ends the episode. If your own portion finishes first, send exactly one coordination_phase=portion_complete message saying only that your part is done and naming one exact release keyword X, for example `My portion is done. Release "X" if you need me again`. On your very next call, call wait_for_signal with about=X. The private wait call is not visible to the partner, so the message must name the same X and say to release it. Stay blocked until the partner sends a later matching release.

Physical workspace handoffs:
- give_space(X) frees X only after that tool call finishes. The partner must not enter or use X in the same concurrent tick.
- An unblocked partner may enter or use X on the following tick without calling wait_for_signal and without receiving a release.
- Use wait_for_signal only when the agent must actually block before X becomes free. Once waiting, it remains blocked throughout a later matching release tick and resumes only on the following tick. Never wait and receive the matching release in the same tick.
Workspace relationships:
- A cabinet and its parent counter are one shared workspace. Navigate to the parent counter for cabinet work. Cabinet tools may be used from that counter, and different agents may use different objects there without a handover. Do not open or close the cabinet during another agent's access to its contents.
- If an agent is at an exclusive child appliance on a parent counter, it may use that counter and objects on it without navigating again. This does not move the agent: its location remains the exclusive child appliance.
The tool schemas are provided separately as function definitions.

Rules:
- Predict exactly one next tool call.
- Use symbolic IDs only, never concrete simulator IDs.
- Symbolic IDs are dictionary keys in the initial state; copy the relevant object, fixture, part, or control key verbatim. Use a site ID only when the state explicitly provides a named site.
- The acting agent is fixed by the prompt; do not choose actions for the other agent.
- Before ANY non-communicate action, BOTH agents must each have sent at least one communicate message; task actions are rejected until then.
- The tool must be one of the provided function schemas.
- Supply exactly the arguments required by the selected tool.
- Prefer the most immediate executable next action.
- Do not emit markdown or narrative after the tool call.
```

### Function/tool schemas (both paths)

```json
[
  {
    "function": {
      "description": "Send a short coordination message to the other agent in the scene. args.to must be that agent's exact ID. Set args.releases to the exact symbolic id of a thing you have finished with and are handing over -- that, and only that, wakes a partner who is waiting on it. Omit args.releases from every other message. args.releases must be an object id or a fixture id that appears in initial_state, copied exactly. It names a THING, never an event or a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', 'counter_free' and 'task_done' are not, and an id that does not exist releases nothing at all. args.coordination_phase is required for protocol messages and must be one of propose, await_plan, await_confirmation, confirm, or portion_complete. Omit it from ordinary requests, corrections, handoffs, and informational messages.",
      "name": "communicate",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "coordination_phase": {
            "description": "Optional opening-protocol phase, or portion_complete when reporting only this agent's assigned work complete. Omit for ordinary messages.",
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
            "description": "Short natural-language coordination message for the other agent.",
            "type": "string"
          },
          "releases": {
            "description": "Optional exact symbolic object or fixture ID being handed over. Include only for a real handover; omit from all other messages.",
            "type": "string"
          },
          "to": {
            "description": "Exact ID of the agent receiving the message.",
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
            "description": "Exact symbolic object or fixture ID being awaited.",
            "type": "string"
          },
          "from": {
            "description": "Exact ID of the agent whose matching release will end the wait.",
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
      "description": "Close a hinged door, lid, or articulated head on a fixture.",
      "name": "close_hinged_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Push in a sliding drawer or rack on a fixture.",
      "name": "close_sliding_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Capture one or more images from named views for later inspection.",
      "name": "get_image",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "views": {
            "description": "One or more camera-view names to request.",
            "items": {
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
  },
  {
    "function": {
      "description": "Leave a workspace when vacating it is genuinely needed so another agent can enter or work there. This is a handover action, not an idle, completion, or participation action. Roomy counters, islands, and dining tables normally need no give_space when agents use different objects.",
      "name": "give_space",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "description": "Exact symbolic ID of the fixture or workspace.",
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
      "description": "Move the robot base to the working pose of a fixture.",
      "name": "navigate_to_fixture",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "fixture_id": {
            "description": "Exact symbolic ID of the fixture or workspace.",
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
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Pull out a sliding drawer or rack on a fixture.",
      "name": "open_sliding_part",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "part_id": {
            "description": "Exact key of the part under initial_state.fixtures[target_id].parts; copy it verbatim (for example door, hinged, lid, or sliding).",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "source_id": {
            "description": "Exact symbolic ID of the object's current source region.",
            "type": "string"
          },
          "source_site_id": {
            "description": "Optional exact sub-site within the source; omit when no named sub-site matters.",
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
      "description": "Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters.",
      "name": "place_in_receptacle",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "receptacle_id": {
            "description": "Exact symbolic ID of the destination receptacle.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "receptacle_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object adjacent to a reference object or nearby fixture on the same support surface. Use reference_object_id only for movable objects and reference_fixture_id only for fixtures. When using reference_fixture_id, the agent may navigate either to that reference fixture or to its declared adjacent_location_id before placing. Requires at least one of: reference_object_id or reference_fixture_id.",
      "name": "place_next_to",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "reference_fixture_id": {
            "description": "Exact symbolic ID of the fixture used as the placement reference.",
            "type": "string"
          },
          "reference_object_id": {
            "description": "Exact symbolic ID of the movable object used as the placement reference.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
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
      "description": "Place an object on top of another movable support object such as a plate or cutting_board.",
      "name": "place_on_object",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "support_object_id": {
            "description": "Exact symbolic ID of the movable object supporting the placed object.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "support_object_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id.",
      "name": "place_on_surface",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "relative_position": {
            "description": "Optional named relative placement; omit unless the goal requires one.",
            "type": "string"
          },
          "support_id": {
            "description": "Exact symbolic ID of the fixture or surface supporting the placement.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "support_id"
        ],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Place an object directly beneath a reference fixture. For dispensers (e.g. coffee machine nozzle, sink faucet) the object is positioned at the dispenser output site; for other fixtures (e.g. wall cabinet) the object is placed on the nearest surface below.",
      "name": "place_under",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "object_id": {
            "description": "Exact symbolic ID of the movable object being manipulated.",
            "type": "string"
          },
          "reference_fixture_id": {
            "description": "Exact symbolic ID of the fixture used as the placement reference.",
            "type": "string"
          },
          "target_site_id": {
            "description": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
            "type": "string"
          }
        },
        "required": [
          "object_id",
          "reference_fixture_id"
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
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Activate a discrete lever-like control on a fixture.",
      "name": "press_lever",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "control_id": {
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
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
      "description": "Set a rotary or handle-like control to an explicit goal state.",
      "name": "set_rotary_control",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "control_id": {
            "description": "Exact key of the control under initial_state.fixtures[target_id].controls; copy it verbatim.",
            "type": "string"
          },
          "goal": {
            "description": "Requested final symbolic state of the control.",
            "type": "string"
          },
          "target_id": {
            "description": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
            "type": "string"
          }
        },
        "required": [
          "target_id",
          "control_id",
          "goal"
        ],
        "type": "object"
      }
    },
    "type": "function"
  }
]
```

### Saved initial state (included identically in both model prompts)

```json
{
  "agents": {
    "agent_0": {
      "held_object": null,
      "location": "cabinet_parent_counter"
    },
    "agent_1": {
      "held_object": null,
      "location": "coffee_machine"
    }
  },
  "fixtures": {
    "cab": {
      "controls": {},
      "fixture_type": "cabinet",
      "parent_fixture": "cabinet_parent_counter",
      "parts": {
        "hinged": {
          "part_type": "hinged_part",
          "state": "closed"
        }
      }
    },
    "cabinet_parent_counter": {
      "controls": {},
      "fixture_type": "counter",
      "parts": {}
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
