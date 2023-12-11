import scipy
import openmdao.api as om


class ActuatorDisc(om.ExplicitComponent):
    """Simple wind turbine model based on actuator disc theory"""

    def setup(self):
        # Inputs
        self.add_input('a', 0.5, desc="Induced Velocity Factor")
        self.add_input('Area', 10.0, units="m**2", desc="Rotor disc area")
        self.add_input('rho', 1.225, units="kg/m**3", desc="air density")
        self.add_input('Vu', 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor")

        # Outputs
        self.add_output('Vr', 0.0, units="m/s",
                        desc="Air velocity at rotor exit plane")
        self.add_output('Vd', 0.0, units="m/s",
                        desc="Slipstream air velocity, downstream of rotor")
        self.add_output('Ct', 0.0, desc="Thrust Coefficient")
        self.add_output('thrust', 0.0, units="N",
                        desc="Thrust produced by the rotor")
        self.add_output('Cp', 0.0, desc="Power Coefficient")
        self.add_output('power', 0.0, units="W", desc="Power produced by the rotor")

        # Every output depends on `a`
        self.declare_partials(of='*', wrt='a', method='cs')

        # Other dependencies , calculate derivative, use the for all modules except dynamics
        self.declare_partials(of='Vr', wrt=['Vu'], method='cs')
        self.declare_partials(of=['thrust', 'power'], wrt=['Area', 'rho', 'Vu'], method='cs')

    def compute(self, inputs, outputs):
        """ Considering the entire rotor as a single disc that extracts
        velocity uniformly from the incoming flow and converts it to
        power."""

        a = inputs['a']
        Vu = inputs['Vu']
        qA = .5 * inputs['rho'] * inputs['Area'] * Vu ** 2

        outputs['Vd'] = Vd = Vu * (1 - 2 * a)
        outputs['Vr'] = .5 * (Vu + Vd)

        outputs['Ct'] = Ct = 4 * a * (1 - a)
        outputs['thrust'] = Ct * qA

        outputs['Cp'] = Cp = Ct * (1 - a)
        outputs['power'] = Cp * qA * Vu

# optimized all four modules
# add_subsystem * 4ww
# create 4 module classes like ActuatorDisc
# var bounds
# prob.model.add_constraint()


prob = om.Problem()
prob.model.add_subsystem('a_disk', ActuatorDisc(), promotes_inputs=['a', 'Area', 'rho', 'Vu'])
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('a', lower=0., upper=1.)

# negative one so we maximize the objective
prob.model.add_objective('a_disk.Cp', scaler=-1)

prob.setup()

prob.set_val('a', .5)
prob.set_val('Area', 10.0, units='m**2')
prob.set_val('rho', 1.225, units='kg/m**3')
prob.set_val('Vu', 10.0, units='m/s')

print(prob.get_val('a_disk.Cp'))
print(prob.get_val('a'))
fail = prob.run_driver()
prob.model.list_inputs(val=True, units=True)

