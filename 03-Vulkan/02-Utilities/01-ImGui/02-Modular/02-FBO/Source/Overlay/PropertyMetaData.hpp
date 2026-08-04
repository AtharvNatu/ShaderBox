#ifndef PROPERTY_META_DATA_HPP
#define PROPERTY_META_DATA_HPP

#include <string>
#include <functional>
#include <chrono>
#include <vector>
#include <memory>
#include <cmath>

#include "imgui.h"

class UIProperty
{
    public:
        virtual ~UIProperty() = default;

        virtual void draw() = 0;

        std::string categoryName;
        std::string label;
        int column = 0;
        std::function<void()> onChanged;

    protected:
        void notify(bool changed)
        {
            if (changed && onChanged)
                onChanged();
        }

};

class UICategory
{
    public:
        std::string name;
        bool expanded = true;
        std::vector<std::unique_ptr<UIProperty>> properties;
};

template<typename T>
class UIValue : public UIProperty
{
    public:
        T* value = nullptr;

        UIValue(
            std::string categoryName, 
            std::string label, 
            T* value, 
            std::function<void()> callback = nullptr
        )
        {
            this->categoryName = std::move(categoryName);
            this->label = std::move(label);
            this->value = value;
            this->onChanged = std::move(callback);
        }

        virtual void draw() override = 0;
};

//! Button
//* ------------------------------------------------------------------------------------------------------------
class UIButton: public UIProperty
{
    public:
        ImVec2 buttonSize;

        UIButton(
            std::string categoryName, 
            std::string label,
            float width = 90.0f,
            float height = 30.0f,
            std::function<void()> callback = nullptr
        )
        {
            this->categoryName = std::move(categoryName);
            this->label = std::move(label);
            this->buttonSize = ImVec2(width, height);
            this->onChanged = std::move(callback);
        }

        void draw() override
        {
            if (ImGui::Button(this->label.c_str(), this->buttonSize))
                notify(true);
        }
};
//* ------------------------------------------------------------------------------------------------------------

//! Text
//* ------------------------------------------------------------------------------------------------------------
class UIText : public UIProperty
{
    public:
        std::string value;
        ImVec4 imColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

        UIText(
            std::string categoryName, 
            std::string value,
            const glm::vec4& color
        )
        {
            this->categoryName = std::move(categoryName);
            this->value = std::move(value);
            imColor = ImVec4(color.r, color.g, color.b, color.a);    
        }

        void draw() override
        {
            ImGui::BeginDisabled();
            ImGui::TextColored(imColor, "%s", this->value.c_str());
            ImGui::EndDisabled();
        }
};

class UIDynamicText : public UIProperty
{
    private:
        std::string cachedText;
        std::chrono::steady_clock::time_point lastUpdate {};
        static constexpr std::chrono::milliseconds updateInterval {250};
        bool primed = false;

    public:
        std::function<std::string()> textCallback;
        ImVec4 imColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

        UIDynamicText(
            std::string categoryName,
            std::function<std::string()> callback,
            const glm::vec4& color = glm::vec4(1.0f)
        )
        {
            imColor = ImVec4(color.r, color.g, color.b, color.a);
            
            this->categoryName = std::move(categoryName);
            this->textCallback = std::move(callback);
        }

        void draw() override
        {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (!primed || (now - lastUpdate) >= updateInterval)
            {
                cachedText = textCallback();
                lastUpdate = now;
                primed = true;
            }

            ImGui::BeginDisabled();
            ImGui::TextColored(imColor, "%s", cachedText.c_str());
            ImGui::EndDisabled();
        }
};
//* ------------------------------------------------------------------------------------------------------------

//! Slider
//* ------------------------------------------------------------------------------------------------------------
class UISliderInt : public UIValue<int>
{
    public:
        int min;
        int max;
        int dimension = 1;

        UISliderInt(
            std::string categoryName, 
            std::string label,
            int* value,
            int min,
            int max,
            int dimension = 1,
            std::function<void()> callback = nullptr
        )
        : UIValue<int>(std::move(categoryName), std::move(label), value, std::move(callback)), 
          min(min),
          max(max),
          dimension(dimension)
        {
        }

        void draw() override
        {
            bool changed = false;

            switch(dimension)
            {
                case 1: changed = ImGui::SliderInt(this->label.c_str(), this->value, this->min, this->max); break;
                case 2: changed = ImGui::SliderInt2(this->label.c_str(), this->value, this->min, this->max); break;
                case 3: changed = ImGui::SliderInt3(this->label.c_str(), this->value, this->min, this->max); break;
                case 4: changed = ImGui::SliderInt4(this->label.c_str(), this->value, this->min, this->max); break;
            }

            notify(changed);
        }
};

class UISliderFloat : public UIValue<float>
{
    public:
        float min;
        float max;
        int dimension = 1;

        UISliderFloat(
            std::string categoryName, 
            std::string label,
            float* value,
            float min,
            float max,
            int dimension = 1,
            std::function<void()> callback = nullptr
        )
        : UIValue<float>(std::move(categoryName), std::move(label), value, std::move(callback)), 
          min(min),
          max(max),
          dimension(dimension)
        {
        }

        void draw() override
        {
            bool changed = false;

            switch(dimension)
            {
                case 1: changed = ImGui::SliderFloat(this->label.c_str(), this->value, this->min, this->max); break;
                case 2: changed = ImGui::SliderFloat2(this->label.c_str(), this->value, this->min, this->max); break;
                case 3: changed = ImGui::SliderFloat3(this->label.c_str(), this->value, this->min, this->max); break;
                case 4: changed = ImGui::SliderFloat4(this->label.c_str(), this->value, this->min, this->max); break;
            }

            notify(changed);
        }
};
//* ------------------------------------------------------------------------------------------------------------

//! CheckBox
//* ------------------------------------------------------------------------------------------------------------
class UICheckBox : public UIValue<bool>
{
    public:

        using UIValue<bool>::UIValue;

        void draw() override
        {
            bool changed = ImGui::Checkbox(this->label.c_str(), this->value);
            notify(changed);
        }
};
//* ------------------------------------------------------------------------------------------------------------

//! Radio Button
//* ------------------------------------------------------------------------------------------------------------
class UIRadioButton: public UIValue<int>
{
    public:
        int data;
        bool sameLine;

        UIRadioButton(
            std::string categoryName, 
            std::string label,
            int* value,
            int data,
            bool sameLine = true,
            std::function<void()> callback = nullptr
        ) : UIValue<int>(std::move(categoryName), std::move(label), value, std::move(callback)),
            data(data),
            sameLine(sameLine)
        {}

        void draw() override
        {
            bool changed = ImGui::RadioButton(this->label.c_str(), this->value, this->data);
            notify(changed);
            if (sameLine)
                ImGui::SameLine();
        }
};
//* ------------------------------------------------------------------------------------------------------------

//! Plot Lines
//* ------------------------------------------------------------------------------------------------------------
class UIPlotLines : public UIProperty
{
    public:
        const std::vector<float>* buffer;
        float scaleMin;
        float scaleMax;
        ImVec2 graphSize;
        std::string hiddenLabel;

        UIPlotLines(
            std::string categoryName,
            std::string label,
            const std::vector<float>* buffer,
            float scaleMin = FLT_MAX,
            float scaleMax = FLT_MAX,
            ImVec2 graphSize = ImVec2(100.0f, 100.0f)
        )
        {
            this->hiddenLabel = "##" + label;
            this->categoryName = std::move(categoryName);
            this->label = std::move(label);
            this->buffer = buffer;
            this->scaleMin = scaleMin;
            this->scaleMax = scaleMax;
            this->graphSize = graphSize;
        }
        
        void draw() override
        {
            ImGui::PlotLines(
                hiddenLabel.c_str(),
                buffer->data(),
                (int)buffer->size(),
                0,
                nullptr,
                scaleMin,
                scaleMax,
                graphSize
            );

            ImVec2 rectMin = ImGui::GetItemRectMin();
            ImVec2 rectMax = ImGui::GetItemRectMax();
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            // Title
            const ImVec2 titlePadding(6.0f, 4.0f);
            ImVec2 titlePos(rectMin.x + titlePadding.x, rectMin.y + titlePadding.y);
            drawList->AddText(titlePos, ImGui::GetColorU32(ImGuiCol_Text), label.c_str());

            // Grid Overlay
            const ImU32 gridColor = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.05f));

            const int gridColumns = 6;
            for (int i = 1; i < gridColumns; i++)
            {
                float x = rectMin.x + (rectMax.x - rectMin.x) * (float)i / gridColumns;
                drawList->AddLine(ImVec2(x, rectMin.y), ImVec2(x, rectMax.y), gridColor);
            }

            const int gridRows = 4;
            for (int i = 1; i < gridRows; i++)
            {
                float y = rectMin.y + (rectMax.y - rectMin.y) * (float)i / gridRows;
                drawList->AddLine(ImVec2(rectMin.x, y), ImVec2(rectMax.x, y), gridColor);
            }
        }

};

//* ------------------------------------------------------------------------------------------------------------


#endif
