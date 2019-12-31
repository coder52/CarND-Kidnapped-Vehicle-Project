/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i=0;i<num_particles;i++){
    Particle particle;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(1);
  }
  is_initialized = true;

  //std::cout<<"1. init() implemented"<<std::flush;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::default_random_engine gen;

   for(int i=0;i<num_particles;i++){

     normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
     normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
     normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
     double x = dist_x(gen);
     double y = dist_y(gen);
     double theta = dist_theta(gen);

     if(fabs(yaw_rate)>0.0001){
       particles[i].x = x + velocity/yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
       particles[i].y = y + velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
       particles[i].theta = theta + yaw_rate * delta_t;
     } else {
       particles[i].x = x + velocity * delta_t * cos(theta);
       particles[i].y = y + velocity * delta_t * sin(theta);
     }
   }
   //std::cout<<"2. prediction() implemented"<<std::flush;
}

//void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

//}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   int num_observation = observations.size();
   int num_landmarks = map_landmarks.landmark_list.size();
   double normalizer;

   for(int j=0;j<num_particles;j++){

     double x_part = particles[j].x;
     double y_part = particles[j].y;
     double theta = particles[j].theta;
     double x_map, y_map;
     int best_landmark;

     for(int i=0;i<num_observation;i++){

       double x_obs = observations[i].x;
       double y_obs = observations[i].y;

       // transform to map x coordinate
       x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);

       // transform to map y coordinate
       y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

       double min_dist = sensor_range;

       for(int k=0;k<num_landmarks;k++){

         double distance = dist(x_map, y_map, map_landmarks.landmark_list[k].x_f,
                                              map_landmarks.landmark_list[k].y_f);

         if(distance < min_dist){
           min_dist = distance;
           best_landmark = map_landmarks.landmark_list[k].id_i;
         }

       }

       particles[j].weight *= multiv_prob(std_landmark[0], std_landmark[1], x_map, y_map,
                             map_landmarks.landmark_list[best_landmark-1].x_f,
                             map_landmarks.landmark_list[best_landmark-1].y_f);
     }

     normalizer += particles[j].weight;
   }

   for (int i = 0; i < num_particles; i++) {

    particles[i].weight /= normalizer;
    weights[i] = particles[i].weight;

  }
  //std::cout<<"3. updateWeights() implemented"<<std::flush;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   std::vector<Particle> new_particles(num_particles);

   int index = rand() % num_particles;
   double beta = 0.0;
   double mw = *max_element(weights.begin(), weights.end());

   for(int i=0; i<num_particles; ++i){

       beta += (rand() / (RAND_MAX + 1.0)) * (2*mw);
       while(beta>weights[index]){
           beta -= weights[index];
           index = (index+1) % num_particles;
       }
       new_particles[i] = particles[index];

    }

    particles = new_particles;

    //std::cout<<"4. resample() implemented"<<std::flush;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
