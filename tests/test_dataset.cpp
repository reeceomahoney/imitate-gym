//
// Created by Reece O'Mahoney on 06/12/22.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "helpers/Utility.hpp"
#include "filesystem"

int main() {
    std::string cwd = std::filesystem::current_path().string();

    raisim::World world;
    auto anymal = world.addArticulatedSystem(cwd + "/resources/models/anymal_c/urdf/model.urdf");
    anymal->setName("anymal");
    anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world.addGround();

    raisim::Vec<3> gravity;
    gravity[0] = 0; gravity[1] = 0; gravity[2] = 0;
    world.setGravity(gravity);

    double control_dt = 0.04, simulation_dt = 0.001;
    world.setTimeStep(simulation_dt);

    /// load dataset
    auto dataset = load_csv<MatrixXd>(cwd + "/resources/expert_data/expert_data_processed.csv");

    /// launch raisim server for visualization. Can be visualized on raisimUnity
    raisim::RaisimServer server(&world);
    server.launchServer(8080);

    Eigen::Matrix<double, 19, 1> gc;
    Eigen::Matrix<double, 18, 1> gv;

    gc << 0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;
    anymal->setState(gc, gv);
//    Eigen::VectorXd pGain(18); pGain.tail(12).setConstant(400);
//    Eigen::VectorXd dGain(18); dGain.tail(12).setConstant(20);
//    anymal->setPdGains(pGain, dGain);

    double t = 0, row = 0;
    while (t < 200) {
        Eigen::VectorXd state = dataset.row(row);

        /// build gc
        gc << 0, 0, state(33), // position
        state(37), state.segment(34, 3), // orientation (quaternions)
        state.segment(3, 12); // joint positions

        /// build gv
        gv << 0, 0, 0, // linear velocity
        state.segment(15, 3), // angular velocity
        state.segment(18, 12); // joint velocities

        anymal->setState(gc, gv);

        for(int i=0; i< int(control_dt / simulation_dt + 1e-10); i++) {
            raisim::MSLEEP(1);
            server.integrateWorldThreadSafe();
        }

        t += control_dt;
        row++;
    }

    server.killServer();
}
