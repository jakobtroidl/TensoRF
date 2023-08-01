import vtk
import numpy as np

# Create a VTK volume from a NumPy array
def create_volume(array):
    # Create a VTK image data object from the NumPy array
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(array.shape)
    imageData.SetSpacing(1, 1, 1)
    imageData.SetOrigin(0, 0, 0)
    imageData.AllocateScalars(vtk.VTK_FLOAT, 1)
    imageData.GetPointData().GetScalars().SetVoidArray(array.ravel(), array.size, 1)

    # Create a VTK volume from the image data
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputData(imageData)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.ShadeOff()
    volumeProperty.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume

if __name__ == '__main__':
    # Create a VTK renderer and window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.2, 0.2)
    renderer.ResetCamera()

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(800, 800)
    renderWindow.AddRenderer(renderer)

    # Create a VTK interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Create a NumPy array to visualize
    array = np.load('/home/jakobtroidl/Desktop/neuralObjects/log/bunny_sdf/volume.npy')

    # Create a VTK volume from the NumPy array
    volume = create_volume(array)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Start the interactor
    interactor.Initialize()
    interactor.Start()