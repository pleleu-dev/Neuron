/**
 * Our activation functions definition, we’ll use classical sigmoid function
 *
 */

import { exp } from "mathjs";

export function sigmoid(x, derivative) {
	let fx = 1 / (1 + exp(-x));
	if (derivative) return fx * (1 - fx);
	return fx;
}
