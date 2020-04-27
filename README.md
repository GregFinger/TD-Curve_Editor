# [Curve Editor](https://vimeo.com/375371794)

This is meant to be a reduced version of the Animation Editor which can work in Perform Mode. Drag the .tox component into any project and it will output a single chop channel of the curve which can be used to drive any animation.

![ui](/Source/img/ui.PNG)

## Usage

Points can be added by holding down alt/option and clicking within the grid.

Points can be moved by clicking and dragging the blue dot. Note: the first and last points are a fixed coordinate on the x-axis, but can be moved up and down on the y-axis.

Points can be deleted by being selected and then pressing the delete key. Note: the first and last points cannot be deleted.

The tab key will cycle through and select each point. This is a non-destructive way of selecting a point, since clicking on a point will often shift its position.

Dragging the yellow dot will adjust the slope of the point.

## UI
**Resolution:** This adjusts the granularity of the curve. Lower values will produce finer curves, but does increase calculation times. This value is clamped between 0.01 and 1. Lower values will give you more samples in your output Chop channel.

**X / Y:** Adjusts the coordinate of the selected point. Clamped between 0 and 1.

**Slope:** Adjusts the slope of the selected point in degrees. Clamped between 0 and 90.

**Type:** Chose between linear and cubic interpolation for the curve.

**Pop Out:** Creates a size-adjustable separate window for the Curve Editor, in case the Curve Editor embedded in your UI is too small to use

## Notes
This is all in python, so calculations could be a little slow. However since this isn't adjusted continuously, there usually is no impact on performance. It's possible to do all the math in GLSL, just involves porting some scipy sub-modules into glsl.

The x and y axis are both currently fixed from 0 to 1. It's possible this can be adjustable in future versions. To adjust it on your own, from the output chop channel, you can use a Math CHOP's range function to remap the y-axis range, or a Stretch CHOP to lengthen or shorten the x-axis.

If points get too close on the x-axis but are too far away on the y-axis, sometimes the results can be noisy. Adjust the point coordinate and slope to solve this.

![ui](/Source/img/error.PNG)

## Future

Enable tangents to have a negative y slope.

Be able to save presets and switch between them.

Snap function and changing grid size

Store ui parameters and points as custom parameters in the component for easy access