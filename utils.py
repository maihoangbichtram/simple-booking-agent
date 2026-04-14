def generate_time_slots(office_timing: str = "9:00-14:00"):
    """Generate hourly time slots from office timing string.

    Args:
        office_timing: String like "11:00-16:00"

    Returns:
        List of time slots like ["11:00 AM", "12:00 PM", ...]
    """
    start_time, end_time = office_timing.split("-")
    start_hour = int(start_time.split(":")[0])
    end_hour = int(end_time.split(":")[0])

    slots = []
    for hour in range(start_hour, end_hour):
        if hour < 12:
            suffix = "AM"
            display_hour = hour if hour > 0 else 12
        elif hour == 12:
            suffix = "PM"
            display_hour = 12
        else:
            suffix = "PM"
            display_hour = hour - 12
        slots.append(f"{display_hour}:00 {suffix}")

    return slots