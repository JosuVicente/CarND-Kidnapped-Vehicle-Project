/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	

	//Initialize number of particles
	num_particles = 150;

	//initialize particles vector
	particles = std::vector<Particle>(num_particles);

	// Create a normal (Gaussian) distribution for x, y and psi
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);


	for (int i = 0; i < num_particles; ++i) {
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1;
	}

	//Initialize weights
	weights.resize(num_particles);
	std::fill(weights.begin(), weights.end(), 1);

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < num_particles; ++i) {
		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
		}
		else {
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y +=  (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		//Add noise
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs closest = LandmarkObs();
		closest.id = 0;
		for (int j = 0; j < predicted.size(); j++) {
			if (closest.id == 0) {
				closest = predicted[j];
			}
			else if (dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) <
				dist(observations[i].x, observations[i].y, closest.x, closest.y)) {
				closest = predicted[j];
			}
		}
		observations[i].id = closest.id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Reset weights vector
	std::fill(weights.begin(), weights.end(), 1);

	//To avoid loop recalculations
	double w_0 = 1 / (2 * M_PI*std_landmark[0] * std_landmark[1]);
	double w_1_x = 2 * pow(std_landmark[0], 2);
	double w_1_y = 2 * pow(std_landmark[1], 2);

	//Transform, Associate and Calculate weight
	for (int i = 0; i < num_particles; i++) {

		//Transformation
		double cos_theta = cos(particles[i].theta);
		double sin_theta = sin(particles[i].theta);

		vector<LandmarkObs> obs_transformed;
		for (int j = 0; j < observations.size(); j++) {
			obs_transformed.push_back(LandmarkObs{ 
											observations[j].id, 
											cos_theta*observations[j].x - sin_theta*observations[j].y + particles[i].x, 
											sin_theta*observations[j].x + cos_theta*observations[j].y + particles[i].y });
		}

		//Predictions
		vector<LandmarkObs> predicted;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			if (fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) < sensor_range &&
				fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) < sensor_range) {
				predicted.push_back(LandmarkObs{
										map_landmarks.landmark_list[j].id_i,
										map_landmarks.landmark_list[j].x_f,
										map_landmarks.landmark_list[j].y_f });
			}
		}		

		//Data association
		dataAssociation(predicted, obs_transformed);

		//Reset weight
		particles[i].weight = 1.0;

		for (int j = 0; j < obs_transformed.size(); j++) {
			for (int k = 0; k < predicted.size(); k++) {
				if (obs_transformed[j].id == predicted[k].id) {
					particles[i].weight *= w_0 * exp(-(pow(predicted[k].x - obs_transformed[j].x, 2) / w_1_x + (pow(predicted[k].y - obs_transformed[j].y, 2) / w_1_y)));
					break;
				}
			}

		}

		weights[i] = particles[i].weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resamples;
	for (int i = 0; i < num_particles; i++)
	{
		resamples.push_back(particles[distribution(gen)]);
	}

	particles = resamples;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
