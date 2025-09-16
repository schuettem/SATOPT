import numpy as np

# Import ffd.py from the same directory
from ffd import FFD

# Import from drag.py
from drag import (
    compute_aoa_and_area,
    load_c_d_lookup_table,
    compute_drag,
)
from body import body_length, body_volume, create_sphere


def genetic_algorithm(
    objective_func,
    bounds,
    pop_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    constraint_funcs=None,
    tournament_size=3,
):
    """
    Genetic Algorithm optimization with mutation and crossover rates.

    Parameters:
        objective_func: Function to minimize
        bounds: List of (min, max) tuples for each parameter
        pop_size: Population size
        generations: Number of generations
        mutation_rate: Probability of mutation (0.0 to 1.0)
        crossover_rate: Probability of crossover (0.0 to 1.0)
        constraint_funcs: List of constraint functions (should return >= 0)
        tournament_size: Size of tournament selection
    """
    n_params = len(bounds)

    # Initialize population
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, n_params),
    )

    def evaluate_individual(individual):
        """Evaluate fitness including penalties for constraint violations"""
        fitness = objective_func(individual)

        # Add penalty for constraint violations
        if constraint_funcs:
            penalty = 0.0
            for constraint in constraint_funcs:
                violation = constraint(individual)
                if violation < 0:  # Constraint violated
                    penalty += abs(violation) * 1000  # Large penalty
            fitness += penalty

        return fitness

    def tournament_selection(population, fitnesses, tournament_size):
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx].copy()

    def crossover(parent1, parent2, crossover_rate):
        """Uniform crossover between two parents"""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        child1, child2 = parent1.copy(), parent2.copy()
        mask = np.random.random(n_params) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        return child1, child2

    def mutate(individual, mutation_rate, bounds):
        """Gaussian mutation with bounds checking"""
        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                # Gaussian mutation with 10% of parameter range as std
                std = (bounds[i][1] - bounds[i][0]) * 0.1
                mutated[i] += np.random.normal(0, std)

                # Clamp to bounds
                mutated[i] = np.clip(mutated[i], bounds[i][0], bounds[i][1])

        return mutated

    # Evolution loop
    best_fitness = float("inf")
    best_individual = None

    for generation in range(generations):
        # Evaluate population
        fitnesses = np.array([evaluate_individual(ind) for ind in population])

        # Track best solution
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx].copy()

        print(
            f"Generation {generation + 1}/{generations}: Best fitness = {best_fitness:.6f}"
        )

        # Create new population
        new_population = []

        # Elitism: keep best individual
        new_population.append(best_individual.copy())

        # Generate rest of population
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Mutation
            child1 = mutate(child1, mutation_rate, bounds)
            child2 = mutate(child2, mutation_rate, bounds)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = np.array(new_population[:pop_size])

    return best_individual, best_fitness


def optimize_satellite(
    bbox_min,
    bbox_max,
    lattice_shape,
    L_max,
    V_min,
    n_iter,
    radius,
    center,
    lookup_table,
    n_phi=20,
    n_theta=20,
):
    """
    Optimize the satellite mesh to minimize drag using Free-Form Deformation (FFD).

    Parameters:
        bbox_min: Minimum bounding box coordinates [x_min, y_min, z_min].
        bbox_max: maximum bounding box coordinates [x_max, y_max, z_max].
        lattice_shape: Tuple (l+1, m+1, n+1) defining the number of control points in each dimension.

    Returns:
        Optimized mesh vertices.
    """
    # Create FFD instance
    ffd = FFD(bbox_min, bbox_max, lattice_shape)

    # Initialize control points to be symmetric across Y=0 plane
    l, m, n = lattice_shape
    for i in range(l):
        for j in range(m):
            for k in range(n):
                j_sym = m - 1 - j
                if j < j_sym:
                    # Make symmetric pairs: same X and Z, opposite Y
                    x_avg = (ffd.P[i, j, k, 0] + ffd.P[i, j_sym, k, 0]) / 2
                    z_avg = (ffd.P[i, j, k, 2] + ffd.P[i, j_sym, k, 2]) / 2
                    y_val = (ffd.P[i, j, k, 1] - ffd.P[i, j_sym, k, 1]) / 2

                    # Set symmetric values
                    ffd.P[i, j, k, 0] = x_avg
                    ffd.P[i, j_sym, k, 0] = x_avg
                    ffd.P[i, j, k, 2] = z_avg
                    ffd.P[i, j_sym, k, 2] = z_avg
                    ffd.P[i, j, k, 1] = abs(y_val)
                    ffd.P[i, j_sym, k, 1] = -abs(y_val)
                elif j == j_sym:
                    # Center plane should have Y=0
                    ffd.P[i, j, k, 1] = 0.0


    # Create original mesh vertices (unit sphere (radius 0.5, centered at (0.5, 0.5, 0.5)))
    org_mesh_vertices, panels = create_sphere(n_phi, n_theta, radius, center)


    # # Constraints: body length and volume
    # def length_constraint(flat_P):
    #     ffd.set_flat_control_points(flat_P)
    #     optimized_vertices = ffd.deform_mesh(org_mesh_vertices)
    #     # Ensure the body length is less than L_max
    #     return L_max - body_length(optimized_vertices)

    # def volume_constraint(flat_P):
    #     ffd.set_flat_control_points(flat_P)
    #     optimized_vertices = ffd.deform_mesh(org_mesh_vertices)
    #     # Ensure the volume is greater than V_min
    #     return body_volume(optimized_vertices, panels) - V_min

    # constraints = [{'type': 'ineq', 'fun': length_constraint},
    #                {'type': 'ineq', 'fun': volume_constraint}]

    def create_symmetric_control_points(reduced_params):
        """
        Create full symmetric control point array from reduced parameters.
        Only optimize the independent parameters and mirror them for symmetry.
        """
        # Reshape to get the original FFD structure
        P_full = ffd.P.copy()
        l, m, n = lattice_shape

        # Map reduced parameters to control points
        param_idx = 0
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    j_sym = m - 1 - j

                    if j < j_sym:  # Only optimize one half
                        # Set the control point from reduced parameters
                        P_full[i, j, k, 0] = reduced_params[param_idx]  # X
                        P_full[i, j, k, 1] = reduced_params[param_idx + 1]  # Y
                        P_full[i, j, k, 2] = reduced_params[param_idx + 2]  # Z

                        # Mirror to symmetric position
                        P_full[i, j_sym, k, 0] = reduced_params[param_idx]  # Same X
                        P_full[i, j_sym, k, 1] = -reduced_params[
                            param_idx + 1
                        ]  # Opposite Y
                        P_full[i, j_sym, k, 2] = reduced_params[param_idx + 2]  # Same Z

                        param_idx += 3
                    elif j == j_sym:  # Center plane
                        # Only optimize X and Z, Y is always 0
                        P_full[i, j, k, 0] = reduced_params[param_idx]  # X
                        P_full[i, j, k, 1] = 0.0  # Y = 0
                        P_full[i, j, k, 2] = reduced_params[param_idx + 1]  # Z

                        param_idx += 2

        return P_full.flatten()

    def objective_symmetric(reduced_params):
        """
        Objective function that works with reduced symmetric parameters.
        """
        # Create full symmetric control points
        full_params = create_symmetric_control_points(reduced_params)

        # Set the control points and deform
        ffd.set_flat_control_points(full_params)
        deformed_vertices = ffd.deform_mesh(org_mesh_vertices)

        # Compute AoA, areas
        aoas, areas = compute_aoa_and_area(panels, deformed_vertices)

        # Compute drag
        drag = compute_drag(aoas, areas, lookup_table)

        return drag

    def length_constraint_symmetric(reduced_params):
        full_params = create_symmetric_control_points(reduced_params)
        ffd.set_flat_control_points(full_params)
        optimized_vertices = ffd.deform_mesh(org_mesh_vertices)
        return L_max - body_length(optimized_vertices)

    def volume_constraint_symmetric(reduced_params):
        full_params = create_symmetric_control_points(reduced_params)
        ffd.set_flat_control_points(full_params)
        optimized_vertices = ffd.deform_mesh(org_mesh_vertices)
        return body_volume(optimized_vertices, panels) - V_min

    # constraints = [{'type': 'ineq', 'fun': length_constraint_symmetric},
    #                {'type': 'ineq', 'fun': volume_constraint_symmetric}]

    # Create reduced initial guess (only independent parameters)
    l, m, n = lattice_shape
    reduced_x0 = []
    for i in range(l):
        for j in range(m):
            for k in range(n):
                j_sym = m - 1 - j

                if j < j_sym:  # Only include one half
                    reduced_x0.extend(
                        [ffd.P[i, j, k, 0], ffd.P[i, j, k, 1], ffd.P[i, j, k, 2]]
                    )
                elif j == j_sym:  # Center plane - only X and Z
                    reduced_x0.extend([ffd.P[i, j, k, 0], ffd.P[i, j, k, 2]])

    reduced_x0 = np.array(reduced_x0, dtype=np.float64)

    # Bounds: limit displacement from initial positions to prevent self-intersection
    bbox_size = np.array(bbox_max) - np.array(bbox_min)
    max_displacement = 0.3 * min(
        bbox_size
    )  # 30% of smallest dimension for more freedom

    # Create bounds relative to initial control point positions for reduced parameters
    bounds = [
        (reduced_x0[i] - max_displacement, reduced_x0[i] + max_displacement)
        for i in range(len(reduced_x0))
    ]

    # Set up an iteration counter in this scope
    iter_slsqp = 0

    # Perform optimization
    def slsqp_callback(reduced_xk):
        nonlocal iter_slsqp
        iter_slsqp += 1
        cost = objective_symmetric(reduced_xk)

        # Check constraint values
        length_val = length_constraint_symmetric(reduced_xk)
        volume_val = volume_constraint_symmetric(reduced_xk)

        # Compute actual values for debugging
        full_params = create_symmetric_control_points(reduced_xk)
        ffd.set_flat_control_points(full_params)
        optimized_vertices = ffd.deform_mesh(org_mesh_vertices)
        actual_length = body_length(optimized_vertices)
        actual_volume = body_volume(optimized_vertices, panels)

        print(f"[SLSQP iter {iter_slsqp:3d}] drag = {cost:.6f}")
        print(
            f"    Length: {actual_length:.2f} (constraint: {length_val:.2f}, limit: {L_max})"
        )
        print(
            f"    Volume: {actual_volume:.2f} (constraint: {volume_val:.2f}, limit: {V_min})"
        )

    # Launch the optimization using Genetic Algorithm
    print("Starting Genetic Algorithm optimization...")

    # Define constraint functions for GA (should return >= 0 for feasible solutions)
    def constraint_funcs_list(reduced_params):
        constraints = []
        constraints.append(length_constraint_symmetric(reduced_params))
        constraints.append(volume_constraint_symmetric(reduced_params))
        return min(constraints)  # Return the most violated constraint

    best_solution, best_fitness = genetic_algorithm(
        objective_func=objective_symmetric,
        bounds=bounds,
        pop_size=30,  # Population size
        generations=n_iter,  # Use n_iter as generations
        mutation_rate=0.15,  # 15% mutation rate
        crossover_rate=0.8,  # 80% crossover rate
        constraint_funcs=[constraint_funcs_list],
        tournament_size=3,
    )

    print(f"GA optimization completed. Best fitness: {best_fitness:.6f}")

    # Set the best solution
    final_full_params = create_symmetric_control_points(best_solution)
    ffd.set_flat_control_points(final_full_params)
    optimized_mesh_vertices = ffd.deform_mesh(org_mesh_vertices)

    return optimized_mesh_vertices, ffd, panels
