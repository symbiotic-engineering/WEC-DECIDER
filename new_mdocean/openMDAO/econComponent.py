import openmdao.api as om
import numpy as np
from omxdsm import write_xdsm

class econComponent(om.ExplicitComponent):

    def setup(self):
        # define inputs
        self.add_input('mass_material', 0)
        self.add_input('M', 0)
        self.add_input('cost_perkg_mult', np.zeros((3,)))
        self.add_input('N_WEC', 0)
        self.add_input('P_elec', 0)
        self.add_input('FCR', 0)
        self.add_input('cost_perN_mult', 0)
        self.add_input('cost_perW_mult', 0)
        self.add_input('F_max', 0)
        self.add_input('P_max', 0)
        self.add_input('efficiency', 0)

        # define outputs
        self.add_output('LCOE', 0)
        self.add_output('capex_design_dep', 0)
        self.add_output('capex', 0)
        self.add_output('opex', 0)
        self.add_output('pto', 0)
        self.add_output('devicestructure', 0)

        self.declare_partials('*', '*', method='fd')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def design_cost_model(self, mass_material, M, cost_perkg_mult, N_WEC,
                          cost_perN_mult, cost_perW_mult, F_max, P_max):
        """
        Compute design-dependent costs: structural, PTO, and total capex.

        Parameters:
            mass_material: float, mass of material
            M: int, material index (1-based in MATLAB, adjust if needed)
            cost_perkg_mult: list or array, material cost multipliers
            N_WEC: int, number of WEC units
            cost_perN_mult: float, force-based cost multiplier
            cost_perW_mult: float, power-based cost multiplier
            F_max: float, max force
            P_max: float, max power

        Returns:
            capex_design_dep: float, total design-dependent capex
            pto: float, PTO cost
            devicestructure: float, structure cost
        """
        # Structural cost per WEC
        alpha_struct = 0.481
        cost_per_kg = (1.64e6 + 1.31e6 * N_WEC ** (-alpha_struct)) / 687000 * cost_perkg_mult[M]
        devicestructure = cost_per_kg * mass_material

        # PTO cost per WEC
        alpha_pto = 0.206
        pto_const = 92593 + 1051 * N_WEC ** (-alpha_pto)
        pto_power = (0.4454 + 0.9099 * N_WEC ** (-alpha_pto)) * P_max * cost_perW_mult
        pto_force = (0.0086 + 0.0118 * N_WEC ** (-alpha_pto)) * F_max * cost_perN_mult
        pto = pto_const + pto_power + pto_force

        # Total design-dependent capex per WEC
        capex_design_dep = devicestructure + pto

        return capex_design_dep, pto, devicestructure

    def LCOE_from_capex_design_power(self, capex_design_dep, N_WEC, P_elec, FCR, efficiency):
        """
        Calculate LCOE, total capex, and opex based on design-dependent costs.

        Parameters:
            capex_design_dep: float, design-dependent capex per WEC
            N_WEC: int, number of WEC units
            P_elec: float, rated power per WEC (W)
            FCR: float, fixed charge rate
            efficiency: float, efficiency factor

        Returns:
            LCOE: float, levelized cost of energy ($/kWh)
            capex: float, total capital expenditure ($)
            opex: float, total operational expenditure ($)
        """
        # Non-design-dependent capex per WEC (RM3 CBS model)
        alpha_non_design = 0.741
        capex_non_design_dep = 12.68e6 * N_WEC ** (-alpha_non_design) + 1.24e6

        # Total capex
        capex_per_wec = capex_design_dep + capex_non_design_dep
        capex = capex_per_wec * N_WEC

        # Opex per WEC
        alpha_opex = 0.5567
        opex_per_wec = 1.193e6 * N_WEC ** (-alpha_opex)
        opex = opex_per_wec * N_WEC

        # Annual Energy Production (AEP)
        hr_per_yr = 8766
        P_avg = N_WEC * P_elec * efficiency  # average power output (W)
        AEP = P_avg * hr_per_yr / 1000  # convert to kWh

        # LCOE calculation
        LCOE = (FCR * capex + opex) / AEP

        return LCOE, capex, opex

    def compute(self, inputs, outputs):
        mass_material = inputs['mass_material'][0]
        M = int(inputs['M'][0])
        cost_perkg_mult = inputs['cost_perkg_mult']
        N_WEC = inputs['N_WEC'][0]
        P_elec = inputs['P_elec'][0]
        FCR = inputs['FCR'][0]  # fixed charge rate
        cost_perN_mult = inputs['cost_perN_mult']
        cost_perW_mult = inputs['cost_perW_mult']
        F_max = inputs['F_max']
        P_max = inputs['P_max']
        efficiency = inputs['efficiency'][0]

        capex_design_dep, pto, devicestructure = self.design_cost_model(
            mass_material, M, cost_perkg_mult, N_WEC, cost_perN_mult, cost_perW_mult, F_max, P_max
        )

        LCOE, capex, opex = self.LCOE_from_capex_design_power(
            capex_design_dep, N_WEC, P_elec, FCR, efficiency
        )


        outputs['LCOE'] = LCOE
        outputs['capex_design_dep'] = capex_design_dep
        outputs['capex'] = capex
        outputs['opex'] = opex
        outputs['pto'] = pto
        outputs['decvicestructure'] = devicestructure
