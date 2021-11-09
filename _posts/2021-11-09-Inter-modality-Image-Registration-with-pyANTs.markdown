---
layout: post
title:  "Inter-modality Image Registration with pyANTs"
date:   2021-11-09 16:07:59 +0200
categories: jekyll update
---
Working with medical images, it's often very convenient to have all the scans in your dataset share the same coordinate system. 
However, this problem called image registration is sometimes challenging and can become hard to implement in a way that actually works in real life.

In this post I'll walk you through a python example for the  registration of head scans from two different 3D modalities (MRI and CT), using ANTs library.


## The Data
Before I jump in to register the CT scan to the MRI scan, let's take a quick look at them.

As I mentioned, in this post I'll use two types of head scans. 
1. Head MRI - [MNI305](http://nist.mni.mcgill.ca/mni-average-brain-305-mri/)
2. Head CT - from the [CQ500 dataset](http://headctstudy.qure.ai/dataset)

Even though each of these scans is a 3D image volume of a human head, they have quite different characteristics in terms of shape and pixel values.

I used [nibabel](https://nipy.org/nibabel/) to load the images:

```python
import nibabel as nib
import numpy as np

def get_data_from_nii_path(nii_data_path: str) -> np.ndarray:
    series_data = nib.load(nii_data_path).get_data()
    # reorder data to (slices, height, width)
    series_data = np.transpose(series_data, [2, 1, 0])
    return series_data

mri_series_path = '/path/to/mri/data/'
ct_series_path = '/path/to/ct/data/'

mri_series = get_data_from_nii_path(mri_series_path)
ct_series = get_data_from_nii_path(ct_series_path)
```

<center><img src="/assets/mri_example_slice.png" alt="Example slice from an MRI scan" height="400"/>  <img src="/assets/ct_example_slice_windowed.png" alt="Example slice from a CT scan" height="400"/></center>
<center>Example slice from a head MRI scan (left) and a CT scan (right)<br/><br/></center>

```python
print(mri_series.shape)
>>> (156, 220, 172)
print(ct_series.shape)
>>> (239, 512, 512)

print(f'MRI: minimum value={mri_series.min()}, maximum value={mri_series.max()}')
>>> MRI: minimum value=-2.695594434759768e-05, maximum value=110.81832135075183
print(f'CT: minimum value={ct_series.min()}, maximum value={ct_series.max()}')
>>> CT: minimum value=-3023, maximum value=3071
```


## Image Registration

Image Registration is the well researched problem of *transforming two images into the same coordinate system*.
Specifically, one of the images defines the target coordinate system and will be called the *fixed image* or *static image*, and the other, *moving image* will be registered to it.

There are a few decisions we have to make when approaching a registration task. 
1. How much are we willing to distort the image? This will determine the degrees of freedom we want in our registration model
2. What similarity metric fits the images we're using?
3. How do we want to perform interpolation for pixels? 
4. Do we want to use the image intensity directly or extract features from the image and use them to perform the registration?

In this example I'll perform an **intensity-based** optimization, and due to the different characteristics of the images, 
I'll allow a **big amount of distortion**, therefore using the ants `'SyN'` transformation. 
For a less accurate registration that would not distort the moving image, I could use the ants `'Affine'` or `'TRSAA'` transformations 
(for more details about types of transformations please check out the documentation notes [here](https://antspy.readthedocs.io/en/latest/registration.html)).  

As for the possible similarity metrics, for most registration tasks, usually either Mutual Information or Cross Correlation will be enough. 
In general, Cross Correlation works well for intra-modality registration, and Mutual Information works well for both intra- and inter-modality registration. 
Here, since the example is inter-modality, I'll use **Mutual Information**.

<center><img src="/assets/overlay_before_reg_bone.png" alt="Example slice overlay of MRI and CT before registration" height="300"/>
<img src="/assets/overlay_before_reg_bone_2.png" alt="Example slice overlay of MRI and CT before registration" height="300"/></center>
<center>Overlay of slices from the CT and the MRI before registration<br/><br/></center>

> **Note**: to create the overlay images I padded the MRI to match the shape of the CT, normalized each image to range [0, 255] (uint8) and then created the overlay image `overlay = np.ubyte(0.7*img1 + 0.3*img2)`

**OK, so let's do this!**
I'll perform the egistration in two steps:
1. Calculating the transformation parameters
2. Applying the parameters to `moving_series`


### Step 0: numpy to ants (and back)

First, I'll define some basic functions for converting numpy data into ants objects and vice versa.

```python
import ants

def get_ants_from_numpy(arr: np.ndarray) -> ants.ANTsImage:
    arr = np.copy(np.transpose(arr, [1, 2, 0]))
    arr = arr.astype(np.float32)
    return ants.from_numpy(arr)


def get_numpy_from_ants(a: ants.ANTsImage) -> np.ndarray:
    arr = a.numpy()
    arr = np.transpose(arr, [2, 0, 1])
    arr = arr.astype(np.int)
    return arr
```

### Step 1: Get transformation parameters

To perform the optimization process to calculate the transformation parameters we need to apply to
`moving_series` to register it to `static_series`, I'll use [ants.registration](https://antspy.readthedocs.io/en/latest/registration.html).

 ```python
 DEFAULT_REGISTRATION_ITERATIONS = (10, 10, 10)
 MI_METRIC = 'mattes'
 SYN_TRANSFORM = 'SyN'
 
 def get_transformation_params(reference_ants: ants.ANTsImage, moving_atns: ants.ANTsImage, registration_iterations: tuple = DEFAULT_REGISTRATION_ITERATIONS,
                               transform_type: str = SYN_TRANSFORM, aff_metric=MI_METRIC) -> dict:
     return ants.registration(fixed=reference_ants, moving=moving_atns, type_of_transform=transform_type, reg_iterations=registration_iterations, aff_metric=aff_metric)
 ```

`ants.registration` returns a dictionary, containing: 
- The `moving_series` warped to be registered to `static_series` (`'warpedmovout'`)
- The `static_series` warped to be registered to `moving_series` (`warpedfixout`)
- The transformation parameters to register `moving_series` to `static_series` (`'fwdtransforms'`)
- The inverse transformation parameters, that is, to register `static_series` to `moving_series` (`'invtransforms'`)

### Step 2: Apply transformation

To apply the transformation parameters to `moving_series`, I'll use [ants.apply_transforms](https://antspy.readthedocs.io/en/latest/registration.html?highlight=apply_transform#ants.apply_transforms). 

```python
INTERPOLATION_TYPE = 'bSpline'

def apply_transformation(reference_ants: ants.ANTsImage, moving_ants: ants.ANTsImage, transformation: dict, interpolator: str = INTERPOLATION_TYPE) -> ants.ANTsImage:
    return ants.apply_transforms(fixed=reference_ants, moving=moving_ants,
                                 transformlist=transformation['fwdtransforms'], interpolator=interpolator)
```

> **Note**: I could take the 'warpedmovout' from the returned dictionary of ants.registration instead of doing this as a separate step, but I like to control the interpolation explicitly.
> 
> This is specifically useful when you want to use the same parameters to register something in addition to the `moving_series`. 
> For example, often we'd want to use the same parameters to register a segmentation mask for the `moving_series` to the same coordinate system. 
> In that case, we'd use `'nearestNeighbor'` or `'genericLabel'` interpolator.

### Step 3: Tying it all together

```python
def register_series(moving_series_array: np.ndarray, reference_series_array: np.ndarray,
                    registration_iterations=DEFAULT_REGISTRATION_ITERATIONS, 
                    transform_type: str = SYN_TRANSFORM, 
                    aff_metric=MI_METRIC) -> (np.ndarray, dict):
    # convert both scans to ants objects
    moving_series_ants = get_ants_from_numpy(moving_series_array)
    reference_series_ants = get_ants_from_numpy(reference_series_array)

    # get transformation parameters
    transformation = get_transformation_params(reference_series_ants, moving_series_ants, 
                                               registration_iterations=registration_iterations, 
                                               transform_type=transform_type, 
                                               aff_metric=aff_metric)

    # apply the transformation
    registered_ants_series = apply_transformation(reference_series_ants, moving_series_ants, 
                                                  transformation, 
                                                  interpolator=INTERPOLATION_TYPE)

    # retrieve registered numpy array
    registered_series_array = get_numpy_from_ants(registered_ants_series)
    return registered_series_array, transformation

registered_series_array, transformation = register_series(moving_series, static_series)
```

After this registration process, `registered_series_array` above, is a new version of `moving_series`, after it was registered to `static_series`.
<center><img src="/assets/overlay_after_reg_bone.png" alt="Example slice overlay of MRI and CT after registration" height="400"/>
<img src="/assets/overlay_after_reg_bone_2.png" alt="Example slice overlay of MRI and CT after registration" height="400"/></center>
<center>Overlay of slices from the CT and the MRI after registration<br/><br/></center>

<center><img src="/assets/ct_example_slice_windowed.png" alt="Example slice from a CT scan" height="400"/>   <img src="/assets/ct_slice_registered.png" alt="Example slice from the registered CT scan" height="400"/></center>
<center>Example slices from the CT scan before (left) and after (right) registration<br/><br/></center>



Note that this means it's shape has changed to match the shape of `static_series`:
```python
print(f'Shape comparison')
print(f'Moving image shape before registration: {moving_series.shape}')
print(f'Target image shape before registration: {static_series.shape}')
print(f'Moving image shape after registration: {registered_series_array.shape}')

>>> Shape comparison 
>>> Moving image shape before registration: (239, 512, 512)
>>> Target image shape before registration: (156, 220, 172)
>>> Moving image shape after registration: (156, 220, 172)
```

## Summary and Pro Tips

I've walked you through a working example of using python with ANTs library in order to register a 3D CT image to a 3D MRI image.

The tricky part in making a registration process actually work on real data will often be choosing the correct parameters for your specific problem:
- Similarity metric
- Type of transformation 
- Iterations (tradeoff of runtime and accuracy)
- Interpolation technique

***Pro tip #1:*** In many cases, it's enough to use low resolution images in order to get the transformation parameters right, 
and it will significantly speed up the optimization process. 
Therefore, you can drastically downsample your images before getting the transformation parameters, and apply the transformation you got on the full resolution image for improved runtime.

***Pro tip #2:*** `ants.registration` function also takes an optional argument `mask`, which is a binary mask over the `static_image` defining the areas of the image the registration will focus on. 
It can be useful to supply such a mask to disregard areas in the image with no relevant information for the registration task (e.g. background objects). 
