/**
 * NeuralNetwork class implementation
 */

import {
	random,
	multiply,
	dotMultiply,
	mean,
	abs,
	subtract,
	transpose,
	add
} from "mathjs";
import * as activation from "./activations";

export class NeuralNetwork {
	constructor(...args) {
		this.input_nodes = args[0]; //number of input neurons
		this.hidden_nodes = args[1]; //number of hidden neurons
		this.output_nodes = args[2]; //number of output neurons

		this.epochs = 50000;
		this.activation = activation.sigmoid;
		this.lr = 0.5; //learning rate
		this.output = 0;

		this.synapse0 = random([this.input_nodes, this.hidden_nodes], -1.0, 1.0); //connections from input layer to hiden
		this.synapse1 = random([this.hidden_nodes, this.output_nodes], -1.0, 1.0); //connections from hidden layer to output
	}

	train(input, target) {
		for (let i = 0; i < this.epochs; i++) {
			//forward
			let input_layer = input; //input data
			let hidden_layer = multiply(input_layer, this.synapse0).map(v =>
				this.activation(v, false)
			); //output of hidden layer neurons (matrix!)
			let output_layer = multiply(hidden_layer, this.synapse1).map(v =>
				this.activation(v, false)
			); // output of last layer neurons (matrix!)

			//backward
			let output_error = subtract(target, output_layer); //calculating error (matrix!)
			let output_delta = dotMultiply(
				output_error,
				output_layer.map(v => this.activation(v, true))
			); //calculating delta (vector!)
			let hidden_error = multiply(output_delta, transpose(this.synapse1)); //calculating of error of hidden layer neurons (matrix!)
			let hidden_delta = dotMultiply(
				hidden_error,
				hidden_layer.map(v => this.activation(v, true))
			); //calculating delta (vector!)

			//gradient descent
			this.synapse1 = add(
				this.synapse1,
				multiply(transpose(hidden_layer), multiply(output_delta, this.lr))
			);
			this.synapse0 = add(
				this.synapse0,
				multiply(transpose(input_layer), multiply(hidden_delta, this.lr))
			);
			this.output = output_layer;

			if (i % 10000 == 0) console.log(`Error: ${mean(abs(output_error))}`);
		}
	}
	predict(input) {
		let input_layer = input;
		let hidden_layer = multiply(input_layer, this.synapse0).map(v =>
			this.activation(v, false)
		);
		let output_layer = multiply(hidden_layer, this.synapse1).map(v =>
			this.activation(v, false)
		);
		return output_layer;
	}
}
