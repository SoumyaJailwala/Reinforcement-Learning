import mujoco
from mujoco import mjx

def test_mjx_compatibility(xml_path):
    """Test what specifically breaks in MJX"""
    try:
        # Load in regular MuJoCo first
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ MuJoCo loading successful")
        
        # Try MJX conversion
        mjx_model = mjx.put_model(mj_model)
        print(f"✓ MJX conversion successful")
        
        # Test basic simulation
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx.step(mjx_model, mjx_data)
        print(f"✓ MJX simulation successful")
        
        return True, "Fully compatible"
        
    except Exception as e:
        print(f"✗ MJX compatibility issue: {e}")
        return False, str(e)

# Test original MetaWorld Sawyer
compatible, message = test_mjx_compatibility("Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach_CLEANTEST.xml")
