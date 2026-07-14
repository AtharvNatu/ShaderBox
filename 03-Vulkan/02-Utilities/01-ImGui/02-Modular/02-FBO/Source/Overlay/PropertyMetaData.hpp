#ifndef PROPERTY_META_DATA_HPP
#define PROPERTY_META_DATA_HPP

#include <string>
#include <functional>

#include "imgui.h"

class UIProperty
{
    public:
        virtual ~UIProperty() = default;

        virtual void draw() = 0;

        std::string categoryName;
        std::string label;
        bool readOnly = false;
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
template<typename T>
class UISlider : public UIValue<T>
{
    public:
        T min;
        T max;

        UISlider(
            const std::string& categoryName, 
            const std::string& label,
            T* value,
            T min,
            T max,
            bool readOnly = false,
            std::function<void()> callback = nullptr
        )
        : UIValue<T>(categoryName, label, value, readOnly, callback), 
          min(min),
          max(max)
        {
        }

        void draw() override;
};

template<>
inline void UISlider<float>::draw()
{
    if (readOnly)
        ImGui::BeginDisabled();

    bool changed = ImGui::SliderFloat(this->label.c_str(), this->value, min, max);
    notify(changed);

    if (readOnly)
        ImGui::EndDisabled();
}

template<>
inline void UISlider<int>::draw()
{
    if (readOnly)
        ImGui::BeginDisabled();

    bool changed = ImGui::SliderInt(this->label.c_str(), this->value, min, max);
    notify(changed);

    if (readOnly)
        ImGui::EndDisabled();
}
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
    public:

        std::string format;
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
            ImGui::BeginDisabled();
            ImGui::TextColored(imColor, "%s", textCallback().c_str());
            ImGui::EndDisabled();
        }
};


//* ------------------------------------------------------------------------------------------------------------

#endif
