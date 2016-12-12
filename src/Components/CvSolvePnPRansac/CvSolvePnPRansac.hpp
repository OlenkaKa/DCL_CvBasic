/*!
 * \file
 * \brief 
 * \author Aleksandra Karbarczyk
 */

#ifndef CVSOLVEPNPRANSAC_HPP_
#define CVSOLVEPNPRANSAC_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include "Types/Objects3D/Object3D.hpp"
#include "Types/HomogMatrix.hpp"
#include "Types/CameraInfo.hpp"

#include <opencv2/core/core.hpp>


namespace Processors {
namespace CvSolvePnPRansac {

/*!
 * \class CvSolvePnPRansac
 * \brief CvSolvePnPRansac processor class.
 *
 * OpenCV solvePnPRansac function wrapper
 */
class CvSolvePnPRansac: public Base::Component {
public:
    /*!
     * Constructor.
     */
    CvSolvePnPRansac(const std::string & name = "CvSolvePnPRansac");

    /*!
     * Destructor
     */
    virtual ~CvSolvePnPRansac();

    /*!
     * Prepare components interface (register streams and handlers).
     * At this point, all properties are already initialized and loaded to 
     * values set in config file.
     */
    void prepareInterface();

protected:

    /*!
     * Connects source to given device.
     */
    bool onInit();

    /*!
     * Disconnect source from device, closes streams, etc.
     */
    bool onFinish();

    /*!
     * Start component
     */
    bool onStart();

    /*!
     * Stop component
     */
    bool onStop();


    // Input data streams
    Base::DataStreamInPtr<Types::Objects3D::Object3D> in_object3d;
    Base::DataStreamIn<Types::CameraInfo> in_camera_info;

    Base::DataStreamIn<cv::Mat> in_rvec;
    Base::DataStreamIn<cv::Mat> in_tvec;
    Base::DataStreamIn<Types::HomogMatrix> in_homog_matrix;

    // Output data streams
    Base::DataStreamOut<Types::HomogMatrix> out_homog_matrix;
    Base::DataStreamOut<cv::Mat> out_rvec;
    Base::DataStreamOut<cv::Mat> out_tvec;

    // Handlers

    // Properties
    Base::Property<bool> useExtrinsicGuess;
    Base::Property<int> iterationsCount;
    Base::Property<float> reprojectionError;
    Base::Property<double> confidence;

    // Handlers
    void onNewObject3D();

};

} //: namespace CvSolvePnPRansac
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("CvSolvePnPRansac", Processors::CvSolvePnPRansac::CvSolvePnPRansac)

#endif /* CVSOLVEPNPRANSAC_HPP_ */
