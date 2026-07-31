#ifndef PROPERTY_META_DATA_HPP
#define PROPERTY_META_DATA_HPP

#include <string>
#include <functional>
#include <chrono>

#include "imgui.h"

class UIProperty
{
    public:
        virtual ~UIProperty() = default;

        virtual void draw() = 0;

        std::string categoryName;
        std::string label;
        bool readOnly = false;
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
            const std::string& categoryName, 
            const std::string& label, 
            T* value, 
            bool readOnly = false, 
            std::function<void()> callback = nullptr
        )
        {
            this->categoryName = categoryName;
            this->label = label;
            this->value = value;
            this->readOnly = readOnly;
            this->onChanged = callback;
        }

        virtual void draw() override = 0;
};

//! Slider
//* ------------------------------------------------------------------------------------------------------------
class UISliderInt : public UIValue<int>
{
    public:
        int min;
        int max;

        UISliderInt(
            const std::string& categoryName, 
            const std::string& label,
            int* value,
            int min,
            int max,
            bool readOnly = false,
            std::function<void()> callback = nullptr
        )
        : UIValue<int>(categoryName, label, value, readOnly, callback), 
          min(min),
          max(max)
        {
        }

        void draw() override
        {
            if (readOnly)
                ImGui::BeginDisabled();

            bool changed = ImGui::SliderInt(this->label.c_str(), this->value, this->min, this->max);
            notify(changed);

            if (readOnly)
                ImGui::EndDisabled();
        }
};

class UISliderFloat : public UIValue<float>
{
    public:
        float min;
        float max;

        UISliderFloat(
            const std::string& categoryName, 
            const std::string& label,
            float* value,
            float min,
            float max,
            bool readOnly = false,
            std::function<void()> callback = nullptr
        )
        : UIValue<float>(categoryName, label, value, readOnly, callback), 
          min(min),
          max(max)
        {
        }

        void draw() override
        {
            if (readOnly)
                ImGui::BeginDisabled();

            bool changed = ImGui::SliderFloat(this->label.c_str(), this->value, this->min, this->max);
            notify(changed);

            if (readOnly)
                ImGui::EndDisabled();
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
            if (readOnly)
                ImGui::BeginDisabled();

            bool changed = ImGui::Checkbox(this->label.c_str(), this->value);
            notify(changed);

            if (readOnly)
                ImGui::EndDisabled();
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
            const std::string& categoryName, 
            const std::string& value,
            const glm::vec4& color
        ) : value(value)
        {
            imColor.x = color.r;
            imColor.y = color.g;
            imColor.z = color.b;
            imColor.w = color.a;
            
            this->categoryName = categoryName;
        }

        void draw() override
        {
            ImGui::BeginDisabled();
            ImGui::TextColored(imColor, this->value.c_str());
            ImGui::EndDisabled();
        }
};

class UIDynamicText : public UIProperty
{
    private:
        std::string cachedText;
        std::chrono::steady_clock::time_point lastUpdate {};
        static constexpr std::chrono::milliseconds updateInterval {250};

    public:
        std::function<std::string()> textCallback;
        ImVec4 imColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

        UIDynamicText(
            const std::string& categoryName, 
            std::function<std::string()> callback,
            const glm::vec4& color = glm::vec4(1.0f)
        )
        {
            imColor = ImVec4(color.r, color.g, color.b, color.a);
            
            this->categoryName = categoryName;
            this->textCallback = std::move(callback);
        }

        void draw() override
        {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (cachedText.empty() || (now - lastUpdate) >= updateInterval)
            {
                cachedText = textCallback();
                lastUpdate = now;
            }

            ImGui::BeginDisabled();
            ImGui::TextColored(imColor, "%s", cachedText.c_str());
            ImGui::EndDisabled();
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

        UIPlotLines(
            const std::string& categoryName,
            const std::string& label,
            const std::vector<float>* buffer,
            float scaleMin = FLT_MAX,
            float scaleMax = FLT_MAX,
            ImVec2 graphSize = ImVec2(100.0f, 100.0f)
        )
        {
            this->buffer = buffer;
            this->scaleMin = scaleMin;
            this->scaleMax = scaleMax;
            this->categoryName = categoryName;
            this->label = label;
            this->graphSize = graphSize;
        }
        

        void draw() override
        {
            std::string hiddenLabel = "##" + label;

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
