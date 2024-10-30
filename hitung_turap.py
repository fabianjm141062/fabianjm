    def plot_pressure_diagram(self, safety_factor):
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Active Pressure Calculation with surcharge and groundwater effect
        depths = [0]
        active_pressures = [0]
        y_offset = 0
        
        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            if y_offset + height > self.groundwater_level:
                gamma -= 9.81
            
            # Calculate active pressure at the base of the layer
            pressure = Ka * gamma * height + (Ka * self.surcharge if y_offset == 0 else 0)
            active_pressures.append(active_pressures[-1] + pressure)
            y_offset += height
            depths.append(y_offset)
        
        # Passive Pressure Calculation only from bottom for passive layer depth
        bottom_depth = self.total_depth
        passive_depths = [bottom_depth, bottom_depth - self.passive_layer['Depth']]
        passive_pressures = [0]
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        
        for depth in passive_depths[:-1]:
            pressure = passive_K * self.passive_layer['Unit Weight'] * (bottom_depth - depth)
            passive_pressures.insert(0, pressure)
        
        max_depth = max(self.total_depth, bottom_depth - self.passive_layer['Depth'])
        interp_depths = np.linspace(0, max_depth, num=100)
        active_interp = interp1d(depths, active_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        passive_interp = interp1d(passive_depths, passive_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        
        # Plot Active and Passive Pressures
        ax.plot(active_interp, interp_depths, label="Active Pressure", color="red")
        ax.plot(passive_interp, interp_depths, label="Passive Pressure", color="blue")

        # Draw Surcharge as Thick Line at Top
        ax.axhline(y=0, color='green', linestyle='-', linewidth=3, label='Surcharge Load')  # Surcharge thick line at top

        # Groundwater Level
        if self.groundwater_level < self.total_depth:
            ax.axhline(y=self.groundwater_level, color='blue', linestyle='--', label='Groundwater Level')

        # Safety Indicator
        if safety_factor >= self.safety_factor_threshold:
            ax.text(0.5, 0.1, "Safe", color="green", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')
        else:
            ax.text(0.5, 0.1, "Not Safe", color="red", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')

        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(0, max_depth)  # Set y-axis to start from 0 and increase downward
        ax.set_title("Combined Active and Passive Earth Pressure Diagram")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
