def global_step_transform(engine, event_name):
    return engine.state.get_event_attrib_value(event_name)
