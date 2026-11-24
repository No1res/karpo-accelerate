"""Opt-in removal of repeated instructions in text-only single-choice requests."""

from karpo_decision.blocks import BlockLayout


class SingleRubricLayout(BlockLayout):
    def __init__(self, encoder, template, rubrics=None, previews=None):
        super().__init__(encoder, template)
        self.rubrics = rubrics
        self.previews = previews or {}

    def shared(self, state, questions, images=(), reader=None):
        if len(questions) == 1 and questions[0].get("type") == "choice" and not images and reader is None:
            instructions = questions[0].get("instructions")
            if instructions in self.previews:
                preview = self.header + self.item + self.text(self.previews[instructions]) + self.newline
                return self.prefix + preview + self.context + self.text(state)
            if self.rubrics is None or instructions in self.rubrics:
                # The branch still contains the complete rubric next to the options.
                return self.prefix + self.context + self.text(state)
        return super().shared(state, questions, images, reader)
