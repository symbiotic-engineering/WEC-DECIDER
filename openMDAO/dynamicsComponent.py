import openmdao.api as om
import numpy as np


class dynamicsComponent(om.ExplicitComponent):

    def setup(self):
        self.add_input("in_params")
        self.add_input("m_float")
        self.add_input("V_d")
        self.add_input("draft")
