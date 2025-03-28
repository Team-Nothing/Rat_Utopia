#include <Arduino.h>
#include <Wire.h>
#include <BM1390GLV.h>

BM1390GLV::BM1390GLV(void)
{
    Address    = BM1390GLV_DEVICE_ADDRESS;
    ModeCtlVal = 0;

    return;
}

BM1390GLV::~BM1390GLV(void)
{
    Address    = 0;
    ModeCtlVal = 0;

    return;
}

String BM1390GLV::get_driver_version(void)
{
    String version;

    version = BM1390GLV_DRIVER_VERSION;

    return (version);
}

int32_t BM1390GLV::init(void)
{
    int32_t result;
    
    result = init_check();
    if (result == BM1390GLV_COMM_OK) {
        (void)init_setting();
    }
    
    return (result);
}

int32_t BM1390GLV::start(void)
{
    int32_t result;
    uint8_t val;
    
    ModeCtlVal &= BM1390GLV_MODE_MASK;

    val = ModeCtlVal | BM1390GLV_MEASUREMENT;
    result = write(BM1390GLV_MODE_CONTROL, &val, sizeof(val));

    return (result);
}

int32_t BM1390GLV::stop(void)
{
    int32_t result;
    uint8_t val;
    
    ModeCtlVal &= BM1390GLV_MODE_MASK;

    val = ModeCtlVal | BM1390GLV_STANDBY;
    result = write(BM1390GLV_MODE_CONTROL, &val, sizeof(val));

    return (result);
}

int32_t BM1390GLV::get_val(uint8_t *raw)
{
    int32_t result;

    result = read(BM1390GLV_PRESSURE_TEMPERATURE, raw, BM1390GLV_ALL_OUT_SIZE);

    return (result);
}

int32_t BM1390GLV::get_val(float *press, float *temp)
{
    int32_t  result;
    uint8_t  raw_data[BM1390GLV_ALL_OUT_SIZE];
    uint32_t raw_press;
    int16_t  raw_temp;

    result = get_val(raw_data);
    if (result == BM1390GLV_COMM_OK) {
        raw_press  = (uint32_t)raw_data[BM1390GLV_PRESS_OUT_MSB] << 16;
        raw_press |= (uint32_t)raw_data[BM1390GLV_PRESS_OUT_LSB] << 8;
        raw_press |= (uint32_t)raw_data[BM1390GLV_PRESS_OUT_XL];

        raw_temp  = (int16_t)raw_data[BM1390GLV_TEMP_OUT_MSB] << 8;
        raw_temp |= (int16_t)raw_data[BM1390GLV_TEMP_OUT_LSB];

        *press = convert_hpa(raw_press);
        *temp  = convert_degree_celsius(raw_temp);
    };

    return (result);
}

int32_t BM1390GLV::init_check(void)
{
    int32_t result;
    uint8_t id[BM1390GLV_ALL_ID_SIZE];

    result = read(BM1390GLV_MANUFACTURER_ID_PART_ID, id, BM1390GLV_ALL_ID_SIZE);
    if (result == BM1390GLV_COMM_OK) {
        if ((id[BM1390GLV_MANUFACTURER_ID] != BM1390GLV_MANUFACTURER_ID_VAL) || (id[BM1390GLV_PART_ID] != BM1390GLV_PART_ID_VAL)) {
            result = BM1390GLV_WAI_ERROR;
        }
    }
    
    return (result);
}

int32_t BM1390GLV::init_setting(void)
{
    int32_t result;
    uint8_t val;
    
    val = BM1390GLV_POWER_DOWN_VAL;
    result = write(BM1390GLV_POWER_DOWN, &val, sizeof(val));
    if (result == BM1390GLV_COMM_OK) {
        val = BM1390GLV_RESET_VAL;
        (void)write(BM1390GLV_RESET, &val, sizeof(val));

        val = BM1390GLV_MODE_CONTROL_VAL;
        (void)write(BM1390GLV_MODE_CONTROL, &val, sizeof(val));

        ModeCtlVal = val;
    }

    return (result);
}

float BM1390GLV::convert_hpa(uint32_t raw_press)
{
    float press;

    press = (float)raw_press / BM1390GLV_COUNTS_PER_HPA;

    return (press);
}

float BM1390GLV::convert_degree_celsius(int16_t raw_temp)
{
    float temp;

    temp = (float)raw_temp / BM1390GLV_COUNTS_PER_DEGREE_CELSIUS;

    return (temp);
}

int32_t BM1390GLV::write(uint8_t reg, uint8_t *data, int32_t size)
{
    int32_t result;

    Wire.beginTransmission(Address);
    (void)Wire.write(reg);
    (void)Wire.write(data, size);
    result = Wire.endTransmission(true);
    if (result == 0) {
        result = BM1390GLV_COMM_OK;
    } else {
        result = BM1390GLV_COMM_ERROR;
    }

    return (result);
}

int32_t BM1390GLV::read(uint8_t reg, uint8_t *data, int32_t size)
{
    int32_t result;
    uint8_t cnt;

    Wire.beginTransmission(Address);
    (void)Wire.write(reg);
    result = Wire.endTransmission(false);
    if (result == 0) {
        (void)Wire.requestFrom((int32_t)Address, size, true);
        cnt = 0;
        while (Wire.available() != 0) {
            data[cnt] = Wire.read();
            cnt++;
        }
        result = BM1390GLV_COMM_OK;
    } else {
        result = BM1390GLV_COMM_ERROR;
    }

    return (result);
}
