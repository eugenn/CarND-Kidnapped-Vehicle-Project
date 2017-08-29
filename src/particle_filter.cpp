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

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    double std_x = std[0];
    double std_y = std[1];
    double std_psi = std[2];

    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_psi);

    num_particles = 100;

    weights.resize(num_particles, 1.0);

    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen);

    for (int i = 0; i < num_particles; i++) {

        Particle p = {
                i,
                sample_x,
                sample_y,
                sample_theta,
                1

        };

        particles.push_back(p);
    }

    is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    const double delta_theta = yaw_rate * delta_t;

    default_random_engine gen;

    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (auto &p : particles) {

        const double theta = p.theta;
        const double noise_x = dist_x(gen);
        const double noise_y = dist_y(gen);
        const double noise_theta = dist_theta(gen);

        // moving straight
        if (fabs(yaw_rate) < 0.001) {
            p.x += velocity * delta_t * cos(theta) + noise_x;
            p.y += velocity * delta_t * sin(theta) + noise_y;
            p.theta += noise_theta;

        } else {
            const double phi = theta + delta_theta;

            p.x += velocity / yaw_rate * (sin(phi) - sin(theta)) + noise_x;
            p.y += velocity / yaw_rate * (cos(theta) - cos(phi)) + noise_y;
            p.theta = phi + noise_theta;

        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {

    for (auto &obs : observations) {
        double min_dist = 1000.0;

        for (unsigned int j = 0; j < predicted.size(); j++) {

            // calculate distance between observed and predicted
            const double distance = dist(obs.x, obs.y, predicted[j].x, predicted[j].y);

            if (min_dist > distance) {
                obs.id = j;
                min_dist = distance;
            }
        }
    }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

    // constants used later for calculating the new weights
    const double stdx = std_landmark[0];
    const double stdy = std_landmark[1];
    const double na = 0.5 / (stdx * stdx);
    const double nb = 0.5 / (stdy * stdy);
    const double d = sqrt(2.0 * M_PI * stdx * stdy);

    for (int i = 0; i < particles.size(); i++) {

        const double px = particles[i].x;
        const double py = particles[i].y;
        const double ptheta = particles[i].theta;

        vector<LandmarkObs> landmarks_in_range;
        vector<LandmarkObs> map_observations;

        // transform observations
        for (auto &obs : observations) {

            const int oid = obs.id;
            const double ox = obs.x;
            const double oy = obs.y;

            const double transformed_x = px + ox * cos(ptheta) - oy * sin(ptheta);
            const double transformed_y = py + oy * cos(ptheta) + ox * sin(ptheta);

            LandmarkObs observation = {
                    oid,
                    transformed_x,
                    transformed_y
            };

            map_observations.push_back(observation);
        }


        // find map landmarks within the sensor range
        for (auto &land : map_landmarks.landmark_list) {

            const int mid = land.id_i;
            const double mx = land.x_f;
            const double my = land.y_f;

            const double dx = mx - px;
            const double dy = my - py;
            const double error = sqrt(dx * dx + dy * dy);

            if (error < sensor_range) {

                LandmarkObs landmark_in_range = {
                        mid,
                        mx,
                        my
                };

                landmarks_in_range.push_back(landmark_in_range);
            }
        }

        // associate landmark in range (id) to landmark observations
        dataAssociation(landmarks_in_range, map_observations);

        // update the particle weights
        double w = 1.0;

        for (auto &map_obs : map_observations) {

            const int oid = map_obs.id;
            const double ox = map_obs.x;
            const double oy = map_obs.y;

            const double predicted_x = landmarks_in_range[oid].x;
            const double predicted_y = landmarks_in_range[oid].y;

            const double dx = ox - predicted_x;
            const double dy = oy - predicted_y;

            const double a = na * dx * dx;
            const double b = nb * dy * dy;
            const double r = exp(-(a + b)) / d;

            w *= r;
        }

        particles[i].weight = w;
        weights[i] = w;
    }

}

void ParticleFilter::resample() {
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> weighted_resample(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        int index = d(gen);
        weighted_resample[i] = particles[index];
    }

    // assign back resampled particles
    particles = weighted_resample;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}