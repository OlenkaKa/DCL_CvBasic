/*!
 * \file
 * \brief
 * \author Aleksandra Karbarczyk
 */

#include <memory>
#include <string>

#include "CvSolvePnPRansac.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace boost;
using Types::HomogMatrix;
using namespace Types::Objects3D;

namespace Processors {
namespace CvSolvePnPRansac {

CvSolvePnPRansac::CvSolvePnPRansac(const std::string & name):
        Base::Component(name),
		iterationsCount("iterationsCount", 100, "iterationsCount"),
		reprojectionError("reprojectionError", 8.0, "reprojectionError"),
		confidence("minInliersCount", 100, "minInliersCount") {
//		flag("flag", "ITERATIVE", std::string("flag")) {
	registerProperty(iterationsCount);
	registerProperty(reprojectionError);
	registerProperty(confidence);
//	registerProperty(flag);
}

CvSolvePnPRansac::~CvSolvePnPRansac() {
}

void CvSolvePnPRansac::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_object3d", &in_object3d);
	registerStream("in_camera_info", &in_camera_info);
	registerStream("out_homog_matrix", &out_homog_matrix);
	registerStream("out_rvec", &out_rvec);
	registerStream("out_tvec", &out_tvec);
	// Register handlers
	registerHandler("onNewObject3D", boost::bind(&CvSolvePnPRansac::onNewObject3D, this));
	addDependency("onNewObject3D", &in_object3d);
	addDependency("onNewObject3D", &in_camera_info);

}

bool CvSolvePnPRansac::onInit() {
	return true;
}

bool CvSolvePnPRansac::onFinish() {
	return true;
}

bool CvSolvePnPRansac::onStop() {
	return true;
}

bool CvSolvePnPRansac::onStart() {
	return true;
}

void CvSolvePnPRansac::onNewObject3D() {
    boost::shared_ptr<Types::Objects3D::Object3D> object3D = in_object3d.read();
    Types::CameraInfo camera_info = in_camera_info.read();

    Mat modelPoints(object3D->getModelPoints());
    Mat imagePoints(object3D->getImagePoints());

    Mat_<double> rvec;
    Mat_<double> tvec;
    Mat_<double> rotationMatrix;

    vector<int> inliers;

    solvePnPRansac(modelPoints, imagePoints, camera_info.cameraMatrix(), camera_info.distCoeffs(), rvec, tvec, false,
				   iterationsCount, reprojectionError, confidence, inliers);
    Rodrigues(rvec, rotationMatrix);

    CLOG(LERROR) << "**** Inliers number: " << inliers.size();
    CLOG(LINFO) << "rvec = " << rvec << "  tvec = " << tvec;

    // Create homogenous matrix.
    cv::Mat pattern_pose = (cv::Mat_<double>(4, 4) <<
            rotationMatrix(0, 0), rotationMatrix(0, 1), rotationMatrix(0, 2), tvec(0),
            rotationMatrix(1, 0), rotationMatrix(1, 1), rotationMatrix(1, 2), tvec(1),
            rotationMatrix(2, 0), rotationMatrix(2, 1), rotationMatrix(2, 2), tvec(2),
            0, 0, 0, 1);

    CLOG(LINFO) << "pattern_pose:\n" << pattern_pose;

    out_rvec.write(rvec.clone());
    out_tvec.write(tvec.clone());
    out_homog_matrix.write(Types::HomogMatrix(pattern_pose.clone()));
}



} //: namespace CvSolvePnPRansac
} //: namespace Processors
