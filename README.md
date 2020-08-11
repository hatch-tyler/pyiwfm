# pyiwfm
---

This project is a python library for working with the California Department of Water Resources (DWR) Integrated Water Flow Model (IWFM) Applications.

---

1. **GroundwaterNodes**
    GroundwaterNodes in an IWFM application are the foundational component for the model geometry. Groundwater Nodes are the only place where
    x-y coordinates exist, so they provide the positional information for the entire model geometry. All other model components either reference 
    nodes directly or are composed of multiple nodes i.e. elements or multiple elements i.e. subregions or element groups.

2. **IWFMNodes**
   IWFMNodes are a container for all GroundwaterNodes contained in an IWFM application. This represents the information contained in the IWFM Nodal
   Configuration File. This class is more likely to be used as a public class than GroundwaterNodes
   

3. **Elements**
   Elements in an IWFM application are composed of either 3 or 4 GroundwaterNodes ordered counter-clockwise. Elements define the computational mesh in the 
   horizontal plane. Elements also identify which subregion the Element is part of. Subregions are used for the calculation of budgets from the 
   simulation results.

4. **IWFMElements**
   IWFMElements are a container for all Elements in an IWFM application. This represents the information contained in the IWFM Element Configuration File.
   This class is more likely to be used as a public class than Elements.

5. **AppGrid**
   AppGrid in an IWFM application is composed of IWFMNodes and IWFMElements. Together, the AppGrid provides the spatial definition of the model elements and
   allows the calculation of element and subregion areas, checking that nodes are provided in counter-clockwise order, and writing spatial information. This class is designed to be used as the public class for obtaining both IWFMNode and IWFMElement information.

6. **IWFMStratigraphy**
   IWFMStratigraphy is a container that defines the computational mesh in the vertical plane. IWFMStratigraphy specifies the number of layers in the model application and
   defines the Ground Surface Elevation (GSE) and layer thicknesses. In IWFM, each layer is composed of two parts, an aquitard (denoted by the prefix 'A') and an aquifer (denoted by the prefix 'L'). GSE and thicknesses are defined at each GroundwaterNode.

7. **StreamNode**
   

8. **StreamReach**
   

9. **StreamSpecifications**
   

10. **Lake**
    Not implemented at this time.

11. **IWFMLakes**
    Not implemented at this time