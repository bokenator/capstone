based on the experiments, come up with the prompts and place them inside of @w4/experiments_imp There should be 8 files in total.
  1. control prompts
  2. schema-constrained prompts
  3. documentation-grounded prompts
  4. tdd-based prompts
  5. schema-constrained + documentation-grounded prompts
  6. schema-constrained + tdd-based prompts
  7. documentation-grounded + tdd-based prompts
  8. schema-constrained + documentation-grounded + tdd-based prompts.

  For each of these files, the prompts will be asking the system to generate three different strategies. a simple one, a medium complexity one, and a very complex one. the prompt will be asking the the LLM agent to generate the exact same strategy, the only thing that will change is the requirements around each generation.

  For now, ONLY come up with the 3 strategies that you think will adequately illustrate our experiments.

  I want the easy strategy to use the from_signal interface of vectorbt, the medium strategy to use the from_order_func(flexible=False) interface, andq the complex strategy to use the from_order_func(flexible=True) interface.

---

now, implement a function called run_experiment (runs a single experiment) in @w4experiments_imp/run_experiment.py that creates a openai agent instance (https://github.com/openai/openai-agents-python) to generate the code based on the prompt. saves the code in @w4/experiments_imp/code, create a new file for each generation with something like c0_simple_[timestamp].py. The agent will then
