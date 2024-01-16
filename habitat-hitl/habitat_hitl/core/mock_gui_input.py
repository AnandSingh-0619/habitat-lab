from habitat_hitl.core.text_drawer import TextOnScreenAlignment


# Create a dynamic mock KeyNS enum
class KeyNS:
    def __getattr__(self, name):
        return None

# Create a dynamic mock KeyNS enum
class MouseNS:
    def __getattr__(self, name):
        return None



class MockLineRender:
    """
    Mock version of DebugLineRender.
    DebugLineRender has a large public interface. Rather than duplicate it, let's just
    allow any method to be called.
    """
    def __getattr__(self, name):
        # This method is called for any attribute not found on the object
        def any_method(*args, **kwargs):
            # This function accepts any arguments and does nothing
            return None
        return any_method

class ConsoleTextDrawer:
    def add_text(
        self,
        text_to_add,
        alignment: TextOnScreenAlignment = TextOnScreenAlignment.TOP_LEFT,
        text_delta_x: int = 0,
        text_delta_y: int = 0,
    ):
        pass
        # print(text_to_add)
