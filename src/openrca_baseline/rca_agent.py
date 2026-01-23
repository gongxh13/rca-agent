from .controller import control_loop

class RCA_Agent:
    def __init__(self, agent_prompt, basic_prompt) -> None:

        self.ap = agent_prompt
        self.bp = basic_prompt

    def run(self, instruction, logger, max_step=25, max_turn=5, callbacks=None):
            
        logger.info(f"Objective: {instruction}")
        prediction, trajectory, prompt = control_loop(instruction, "", self.ap, self.bp, logger=logger, max_step=max_step, max_turn=max_turn, callbacks=callbacks)
        logger.info(f"Result: {prediction}")

        return prediction, trajectory, prompt